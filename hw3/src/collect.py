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
import time
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
    """Запрашивает одну страницу API, уважая ``backoff`` и временные ошибки."""
    url = f"{API_URL}{path}?{urlencode(query, doseq=True)}"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "mlops-hw3-android-dataset/1.0"})
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed HTTPS host
                payload = json.loads(response.read().decode("utf-8"))
            if "error_id" in payload:
                raise StackExchangeError(payload.get("error_message", "ошибка Stack Exchange API"))
            if payload.get("backoff"):
                time.sleep(int(payload["backoff"]))
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, StackExchangeError) as exc:
            last_error = exc
            if isinstance(exc, StackExchangeError):
                break
            time.sleep(2**attempt)
    raise StackExchangeError(f"не удалось получить {url}: {last_error}")


def fetch_questions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Получает вопросы с принятым ответом до требуемого объёма.

    У Stack Exchange несколько тегов в одном ``tagged`` означают пересечение,
    что для Android-среза почти пусто. Поэтому запрашиваем каждый тег отдельно,
    затем устраняем пересечения по ID вопроса.
    """
    start = int(datetime.fromisoformat(cfg["from_date"]).replace(tzinfo=UTC).timestamp())
    end = int(
        (datetime.fromisoformat(cfg["to_date"]).replace(tzinfo=UTC) + timedelta(days=1)).timestamp()
    ) - 1
    found: dict[int, dict[str, Any]] = {}
    for tag in cfg["tags"]:
        tag_rows = 0
        page = 1
        while tag_rows < cfg["per_tag_rows"]:
            payload = request_json(
                "/search/advanced",
                {
                    "site": cfg["site"],
                    "tagged": tag,
                    "fromdate": start,
                    "todate": end,
                    "accepted": "true",
                    "closed": "false",
                    "sort": "creation",
                    "order": "asc",
                    "pagesize": 100,
                    "page": page,
                    "filter": "withbody",
                },
                cfg["timeout_seconds"],
            )
            items = payload.get("items", [])
            if not items:
                break
            found.update({item["question_id"]: item for item in items})
            tag_rows += len(items)
            if not payload.get("has_more"):
                break
            page += 1
    questions = sorted(found.values(), key=lambda item: item["question_id"])
    if len(questions) < cfg["n_rows"]:
        raise StackExchangeError(
            f"по заданным тегам и периоду найдено только {len(questions)} вопросов; "
            "расширьте collect.versions.<версия>.tags или период"
        )
    return questions[: cfg["n_rows"]]


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


def topic_for(tags: list[str]) -> str:
    """Группа — сигнатура подсистемы, а не общий тег ``android``."""
    ignored = {"android", "kotlin", "java"}
    specific = sorted(tag for tag in tags if tag not in ignored)
    return " + ".join(specific[:3]) if specific else "android-general"


def make_record(question: dict[str, Any], answer: dict[str, Any], prompts: list[str]) -> dict[str, Any] | None:
    question_text = html_to_text(question.get("body", ""))
    answer_text = html_to_text(answer.get("body", ""))
    title = html_to_text(question.get("title", ""))
    if not title or not question_text or not answer_text:
        return None
    example_id = f"so-{question['question_id']}"
    return {
        "id": example_id,
        "topic": topic_for(question.get("tags", [])),
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
    questions = fetch_questions(cfg)
    answers = fetch_accepted_answers(questions, cfg)

    records: list[dict[str, Any]] = []
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
        "from_date": cfg["from_date"],
        "to_date": cfg["to_date"],
        "tags": cfg["tags"],
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
