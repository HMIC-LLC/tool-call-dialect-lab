#!/usr/bin/env python3
"""Score saved, human-labeled tool-call text against parser strictness levels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tool_dialect import LEVELS, parse_detail


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError(f"line {line_number}: missing case id")
        if case["id"] in seen:
            raise ValueError(f"line {line_number}: duplicate case id")
        seen.add(case["id"])
        if not isinstance(case.get("response_text"), str):
            raise ValueError(f"line {line_number}: response_text must be a string")
        if not isinstance(case.get("expected_calls"), list):
            raise ValueError(f"line {line_number}: expected_calls must be a list")
        for call in case["expected_calls"]:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError(f"line {line_number}: expected call must have name and arguments")
            if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
                raise ValueError(f"line {line_number}: expected call has invalid types")
        if case.get("offered_tools") is not None and not isinstance(case["offered_tools"], list):
            raise ValueError(f"line {line_number}: offered_tools must be a list")
        cases.append(case)
    if not cases:
        raise ValueError("no labeled cases")
    return cases


def score(cases: list[dict], source_sha256: str) -> dict:
    rows = []
    for case in cases:
        for level in LEVELS:
            result = parse_detail(case["response_text"], tools=case.get("offered_tools"),
                                  native=case.get("native_tool_calls"), level=level)
            actual = [{"name": call.name, "arguments": call.arguments} for call in result.calls]
            rows.append({"id": case["id"], "level": level, "expected_calls": case["expected_calls"],
                         "actual_calls": actual, "exact_match": actual == case["expected_calls"],
                         "dialects": result.dialects, "flags": result.flags,
                         "refused_reasons": [reason for reason, _ in result.refused]})
    return {"schema": "tool-call-dialect-lab.v1", "source_sha256": source_sha256,
            "cases": len(cases), "summary": {level: {"exact_matches": sum(row["exact_match"] for row in rows if row["level"] == level),
                                                 "total": len(cases)} for level in LEVELS},
            "results": rows,
            "interpretation": "Parser agreement with saved labels only; not a model capability score or causal quantization result."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="UTF-8 JSONL with human-labeled expected_calls")
    parser.add_argument("--output", type=Path, help="optional report file; defaults to stdout")
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    report = score(load_cases(args.cases), hashlib.sha256(raw).hexdigest())
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
