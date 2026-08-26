"""Run the RAG quality eval set against a live RTFM agent instance.

Usage:
    python evals/run_evals.py [--url http://localhost:8000] [--tenant evals]
                              [--dataset evals/datasets/sample_questions.jsonl]

The target server must be running (uvicorn rtfm_agent.api:app) with the
corpus ingested for the tenant under test. Exits non-zero when any example
fails, so it slots into CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.metrics.rag_quality import score_example  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(",")
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000",
                        help="base URL of the running RTFM agent")
    parser.add_argument("--tenant", default="evals",
                        help="X-Tenant-Id to evaluate against")
    parser.add_argument("--dataset", default="evals/datasets/sample_questions.jsonl")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    print(f"Eval set: {len(dataset)} examples against {args.url} "
          f"(tenant={args.tenant})\n")

    passed = 0
    failures: list[dict] = []
    headers = {"X-Tenant-Id": args.tenant}

    with httpx.Client(timeout=120, headers=headers) as client:
        for row in dataset:
            try:
                resp = client.post("/ask", json={"question": row["question"]})
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPError as exc:
                print(f"[{row['id']}] HTTP failure: {exc}")
                failures.append({"id": row["id"], "error": str(exc)})
                continue
            verdict = score_example(row, result)
            mark = "PASS" if verdict["passed"] else "FAIL"
            print(f"[{mark}] {row['id']} {row['question'][:60]}")
            for check in verdict["checks"]:
                flag = "+" if check["passed"] else "-"
                print(f"      [{flag}] {check['name']}: {check['detail']}")
            if verdict["passed"]:
                passed += 1
            else:
                failures.append(verdict)

    total = len(dataset)
    print(f"\nResult: {passed}/{total} passed "
          f"({passed / total:.0%})" if total else "empty dataset")
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
