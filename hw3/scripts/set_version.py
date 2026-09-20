#!/usr/bin/env python3
"""Переключить collect.version в params.yaml, не потеряв комментарии.

yaml.safe_load + yaml.dump сохранил бы данные и выбросил комментарии,
а params.yaml для студента наполовину состоит из объяснений. Поэтому
правится ровно одна строка регулярным выражением.
"""

import re
import sys
from pathlib import Path

PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"
LINE = re.compile(r'^(\s*version:\s*)"[^"]*"(.*)$', re.M)


def main() -> int:
    if len(sys.argv) != 2:
        print("использование: set_version.py v1|v2", file=sys.stderr)
        return 2
    version = sys.argv[1]
    text = PARAMS.read_text(encoding="utf-8")
    if f"\n    {version}:\n" not in text:
        print(f"в collect.sources нет версии {version!r}", file=sys.stderr)
        return 1
    new, count = LINE.subn(rf'\g<1>"{version}"\g<2>', text, count=1)
    if count != 1:
        print("не найдена строка collect.version в params.yaml", file=sys.stderr)
        return 1
    PARAMS.write_text(new, encoding="utf-8")
    print(f"collect.version = {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
