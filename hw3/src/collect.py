"""Стадия collect: Stack Exchange API → ``data/raw.jsonl``.

Набор собирается из первичных публикаций Stack Overflow по Android-разработке,
а не скачивается в виде готового SFT-датасета. Каждая запись содержит реальный
вопрос с принятым ответом; идентификатор вопроса позволяет восстановить
первоисточник как ``https://stackoverflow.com/questions/<id>``.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.config import load_params

API_URL = "https://api.stackexchange.com/2.3"
SOURCE_URL = "https://stackoverflow.com/questions/{question_id}"
HTTP_CACHE = Path(".cache/stackexchange/http")
RECORD_CACHE = Path(".cache/stackexchange/records")
_LAST_REQUEST_AT = 0.0


class StackExchangeError(RuntimeError):
    """API не отдала данные, необходимые для воспроизводимой выгрузки."""


class TextExtractor(HTMLParser):
    """Преобразует HTML поста в текст, сохраняя абзацы и блоки кода."""

    BLOCKS = {"p", "br", "li", "pre", "blockquote", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


def html_to_text(value: str) -> str:
    parser = TextExtractor()
    parser.feed(value)
    parser.close()
    return html.unescape(parser.text())


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбирает системный промпт по идентификатору вопроса."""
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def request_json(path: str, query: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Запрашивает страницу API с локальным кэшем и обработкой лимитов."""
    global _LAST_REQUEST_AT
    api_key = os.environ.get("STACKEXCHANGE_KEY")
    if api_key:
        query = {**query, "key": api_key}
    url = f"{API_URL}{path}?{urlencode(query, doseq=True)}"
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = HTTP_CACHE / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    last_error: Exception | None = None
    for attempt in range(8):
        try:
            # Небольшая пауза не даёт параллельным по смыслу запросам упереться
            # в динамический throttling Stack Exchange даже без API-ключа.
            delay = 0.2 - (time.monotonic() - _LAST_REQUEST_AT)
            if delay > 0:
                time.sleep(delay)
            request = Request(url, headers={"User-Agent": "mlops-hw3-android-dataset/1.0"})
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed HTTPS host
                payload = json.loads(response.read().decode("utf-8"))
            _LAST_REQUEST_AT = time.monotonic()
            if "error_id" in payload:
                raise StackExchangeError(payload.get("error_message", "ошибка Stack Exchange API"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache_path)
            if payload.get("backoff"):
                time.sleep(int(payload["backoff"]))
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, StackExchangeError) as exc:
            last_error = exc
            if isinstance(exc, StackExchangeError):
                break
            if isinstance(exc, HTTPError) and exc.code == 429:
                retry_after = int(exc.headers.get("Retry-After", 10 * 2**attempt))
                time.sleep(min(retry_after, 60))
            else:
                time.sleep(2**attempt)
    raise StackExchangeError(f"не удалось получить {url}: {last_error}")


def fetch_questions(
    cfg: dict[str, Any],
    *,
    seen_ids: set[int] | None = None,
    group_counts: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    """Собирает сбалансированную выборку по фиксированной таксономии.

    Окна обрабатываются последовательно. Поэтому первое окно ``v2`` полностью
    совпадает с ``v1``, а второе добавляет следующие записи того же среза.
    Вопрос, найденный несколькими запросами, закрепляется за первой группой.
    """
    found: list[dict[str, Any]] = []
    seen_ids = set() if seen_ids is None else seen_ids
    group_counts = Counter() if group_counts is None else group_counts

    for window in cfg["windows"]:
        start = int(datetime.fromisoformat(window["from_date"]).replace(tzinfo=UTC).timestamp())
        end = int(
            (
                datetime.fromisoformat(window["to_date"]).replace(tzinfo=UTC)
                + timedelta(days=1)
            ).timestamp()
        ) - 1
        target = window["rows_per_topic"]

        for topic, spec in cfg["topics"].items():
            added = 0
            page = 1
            while added < target and page <= cfg["max_pages_per_topic"]:
                payload = request_json(
                    "/search/advanced",
                    {
                        "site": cfg["site"],
                        "tagged": ";".join(spec["tags"]),
                        "fromdate": start,
                        "todate": end,
                        "accepted": "true",
                        "closed": "false",
                        "sort": "creation",
                        "order": "asc",
                        "pagesize": cfg["candidates_per_page"],
                        "page": page,
                        "filter": "withbody",
                    },
                    cfg["timeout_seconds"],
                )
                items = payload.get("items", [])
                if not items:
                    break
                for item in items:
                    question_id = item["question_id"]
                    if question_id in seen_ids:
                        continue
                    item["_dataset_topic"] = topic
                    found.append(item)
                    seen_ids.add(question_id)
                    group_counts[topic] += 1
                    added += 1
                    if added >= target:
                        break
                if not payload.get("has_more"):
                    break
                page += 1

            minimum = window.get("min_rows_per_topic", 0)
            if added < minimum:
                raise StackExchangeError(
                    f"группа {topic!r}: в окне {window['from_date']}–{window['to_date']} "
                    f"собрано {added}, требуется минимум {minimum}"
                )

    return found


def cache_signature(base: dict[str, Any], version: str) -> str:
    """Идентификатор именно той конфигурации, из которой получены записи."""
    payload = {
        "site": base["site"],
        "version": base["versions"][version],
        "topics": base["topics"],
        "system_prompts": base["system_prompts"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_record_cache(base: dict[str, Any], version: str) -> list[dict[str, Any]] | None:
    """Читает проверенный снимок записей; чужая конфигурация не принимается."""
    records_path = RECORD_CACHE / f"{version}.jsonl"
    metadata_path = RECORD_CACHE / f"{version}.json"
    if not records_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != cache_signature(base, version):
        return None
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    if metadata.get("rows") != len(records):
        return None
    return records


def write_record_cache(base: dict[str, Any], version: str, records: list[dict[str, Any]]) -> None:
    """Сохраняет успешную выгрузку атомарно для продолжения после HTTP 429."""
    RECORD_CACHE.mkdir(parents=True, exist_ok=True)
    records_path = RECORD_CACHE / f"{version}.jsonl"
    temporary = records_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(records_path)
    metadata = {"signature": cache_signature(base, version), "rows": len(records)}
    (RECORD_CACHE / f"{version}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def find_cached_prefix(
    base: dict[str, Any], version: str
) -> tuple[str | None, list[dict[str, Any]], int]:
    """Находит наиболее длинную уже собранную версию, входящую в новую."""
    target_windows = base["versions"][version]["windows"]
    candidates = sorted(
        (
            (name, spec["windows"])
            for name, spec in base["versions"].items()
            if name != version and len(spec["windows"]) < len(target_windows)
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for name, windows in candidates:
        if target_windows[: len(windows)] != windows:
            continue
        records = read_record_cache(base, name)
        if records is not None:
            return name, records, len(windows)
    return None, [], 0


def fetch_accepted_answers(
    questions: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Получает только ответы, помеченные авторами вопросов как принятые."""
    answer_ids = [question["accepted_answer_id"] for question in questions]
    answers: dict[int, dict[str, Any]] = {}
    for start in range(0, len(answer_ids), 100):
        ids = ";".join(str(value) for value in answer_ids[start : start + 100])
        payload = request_json(
            f"/answers/{ids}",
            {"site": cfg["site"], "filter": "withbody", "pagesize": 100},
            cfg["timeout_seconds"],
        )
        answers.update({item["answer_id"]: item for item in payload.get("items", [])})
    return answers


def make_record(question: dict[str, Any], answer: dict[str, Any], prompts: list[str]) -> dict[str, Any] | None:
    question_text = html_to_text(question.get("body", ""))
    answer_text = html_to_text(answer.get("body", ""))
    title = html_to_text(question.get("title", ""))
    if not title or not question_text or not answer_text:
        return None
    example_id = f"so-{question['question_id']}"
    return {
        "id": example_id,
        "topic": question["_dataset_topic"],
        "messages": [
            {"role": "system", "content": pick_prompt(example_id, prompts)},
            {
                "role": "user",
                "content": f"Проблема Android-разработки: {title}\n\n{question_text}",
            },
            {"role": "assistant", "content": answer_text},
        ],
    }


def main() -> None:
    params = load_params()
    base = params["collect"]
    version = base["version"]
    versions = base["versions"]
    if version not in versions:
        raise SystemExit(f"неизвестная collect.version {version!r}; доступны {sorted(versions)}")
    cfg = {**base, **versions[version]}
    prompts = base["system_prompts"]
    if len(prompts) < 3:
        raise SystemExit("collect.system_prompts должен содержать минимум три варианта")

    started = time.perf_counter()
    cached_records = read_record_cache(base, version)
    cache_base_version: str | None = version if cached_records is not None else None
    skipped_windows = len(cfg["windows"]) if cached_records is not None else 0
    if cached_records is None:
        cache_base_version, cached_records, skipped_windows = find_cached_prefix(base, version)

    records: list[dict[str, Any]] = list(cached_records or [])
    seen_ids = {int(record["id"].removeprefix("so-")) for record in records}
    group_counts = Counter(record["topic"] for record in records)
    delta_cfg = {**cfg, "windows": cfg["windows"][skipped_windows:]}
    questions = fetch_questions(delta_cfg, seen_ids=seen_ids, group_counts=group_counts)
    answers = fetch_accepted_answers(questions, cfg)

    dropped_missing_answer = dropped_empty = 0
    for question in questions:
        answer = answers.get(question["accepted_answer_id"])
        if answer is None:
            dropped_missing_answer += 1
            continue
        record = make_record(question, answer, prompts)
        if record is None:
            dropped_empty += 1
            continue
        records.append(record)

    write_record_cache(base, version, records)

    out = Path(params["paths"]["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "version": version,
        "source": "Stack Exchange API v2.3 / Stack Overflow",
        "site": cfg["site"],
        "source_url_pattern": SOURCE_URL,
        "license": "CC BY-SA (версия зависит от даты публикации)",
        "windows": cfg["windows"],
        "configured_topics": len(cfg["topics"]),
        "rows_per_topic": dict(sorted(Counter(record["topic"] for record in records).items())),
        "cache_base_version": cache_base_version,
        "records_reused_from_cache": len(cached_records or []),
        "questions_fetched": len(questions),
        "rows_written": len(records),
        "dropped_missing_accepted_answer": dropped_missing_answer,
        "dropped_empty_post": dropped_empty,
        "groups": len({record["topic"] for record in records}),
        "system_prompt_variants": len({record["messages"][0]["content"] for record in records}),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(params["paths"]["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: {version}, получено вопросов {len(questions)}, записано {len(records)}, "
        f"групп {metrics['groups']}, {metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()
