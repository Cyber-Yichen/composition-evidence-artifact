from __future__ import annotations

import json
import re
import tomllib
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
}
IDENTITY_TERMS = (
    "/" + "Users/",
    "/" + "root/",
    "Yic" + "hen",
    "Yu" + "xi",
    "co-authored" + "-by",
    "internal " + "spec",
    "reviewer" + " will ask",
    "reviewers" + " will ask",
)
IDENTITY_OR_PATH = re.compile(
    "|".join(re.escape(term) for term in IDENTITY_TERMS)
    + r"|/" + r"home/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)
FOREIGN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
SECRET = re.compile(
    r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,}|"
    r"sk-[0-9A-Za-z_-]{20,}|xox[baprs]-[0-9A-Za-z-]{10,}|"
    r"gh[pousr]_[0-9A-Za-z]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"Bearer [0-9A-Za-z._-]{20,}",
)


def audit_text() -> int:
    checked = 0
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        checks = [
            ("identity or machine-specific path", IDENTITY_OR_PATH),
            ("credential-like value", SECRET),
        ]
        # Raw model responses are scientific records and must not be translated.
        if "results" not in path.parts:
            checks.append(("non-English release text", FOREIGN))
        for label, pattern in checks:
            if pattern.search(text):
                raise AssertionError(
                    f"{path.relative_to(REPO)} contains {label}")
        checked += 1
    return checked


def audit_configs() -> int:
    parsed = 0
    for path in sorted((REPO / "configs").glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
        parsed += 1
    for path in sorted((REPO / "configs").glob("*.yaml")):
        yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed += 1
    with (REPO / "pyproject.toml").open("rb") as stream:
        tomllib.load(stream)
    return parsed + 1


def audit_jsonl() -> tuple[int, int]:
    files = 0
    records = 0
    for path in sorted((REPO / "results").rglob("*.jsonl")):
        seen = set()
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("uid")
            if not isinstance(uid, str) or not uid:
                raise AssertionError(
                    f"{path.relative_to(REPO)}:{lineno} lacks a valid UID")
            if uid in seen:
                raise AssertionError(
                    f"{path.relative_to(REPO)} has duplicate UID {uid}")
            seen.add(uid)
            if "evidence" in path.parts:
                required = {"features", "evidence_text"}
            else:
                required = {"correct"}
            missing = required - row.keys()
            if missing:
                raise AssertionError(
                    f"{path.relative_to(REPO)}:{lineno} lacks {sorted(missing)}")
            records += 1
        files += 1
    return files, records


def audit_tree() -> int:
    forbidden = []
    for path in REPO.rglob("*"):
        if path.is_symlink() or path.name in {
                ".DS_Store", "__pycache__", ".pytest_cache"}:
            forbidden.append(str(path.relative_to(REPO)))
        if path.suffix == ".pyc":
            forbidden.append(str(path.relative_to(REPO)))
    if forbidden:
        raise AssertionError(f"forbidden release artifacts: {forbidden}")
    checkpoint = REPO / "results" / "projector" / "best.pt"
    if not checkpoint.is_file() or not zipfile.is_zipfile(checkpoint):
        raise AssertionError("projector checkpoint is missing or malformed")
    return sum(path.is_file() for path in REPO.rglob("*"))


def main() -> None:
    file_count = audit_tree()
    text_count = audit_text()
    config_count = audit_configs()
    jsonl_files, record_count = audit_jsonl()
    print(
        f"Release audit passed: {file_count} files, {text_count} text files, "
        f"{config_count} parsed configs, and {record_count} records across "
        f"{jsonl_files} JSONL files.")


if __name__ == "__main__":
    main()
