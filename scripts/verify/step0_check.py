"""Step 0 verification: Redis Stack (vector search), LLM (OpenAI-compatible), fastembed embeddings."""

import os
import sys

import httpx
from dotenv import load_dotenv
from fastembed import TextEmbedding
from redis import Redis

load_dotenv()

PASS = 0
FAIL = 0

FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash"]


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f" - {detail}" if detail else ""))


def check_redis() -> None:
    r = Redis.from_url(os.getenv("settings.redis.url", "redis://localhost:6379"), decode_responses=True)
    try:
        check("Redis PING", r.ping() is True)
        indexes = r.execute_command("FT._LIST")
        check("FT._LIST reachable", isinstance(indexes, list), f"existing indexes: {indexes}")
        r.execute_command(
            "FT.CREATE",
            "step0_check_idx",
            "ON", "HASH",
            "PREFIX", "1", "step0:check:",
            "SCHEMA",
            "text", "TEXT",
            "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32", "DIM", "4", "DISTANCE_METRIC", "COSINE",
        )
        vec = b"\x00\x00\x80\x3f" * 4  # [1.0, 1.0, 1.0, 1.0] - non-zero, avoids NaN cosine
        r.hset("step0:check:1", mapping={"text": "hello redis", "embedding": vec})
        res = r.execute_command(
            "FT.SEARCH", "step0_check_idx", "*=>[KNN 1 @embedding $B]",
            "PARAMS", "2", "B", vec, "RETURN", "1", "text", "DIALECT", "2",
        )
        total = res["total_results"] if isinstance(res, dict) else res[0]
        check("Vector KNN search", total == 1, f"{total} doc(s) found")
    finally:
        try:
            r.execute_command("FT.DROPINDEX", "step0_check_idx", "DD")
        except Exception:
            pass
        r.close()


def check_llm() -> None:
    from rtfm_agent.config import settings

    base = settings.llm.base_url + "/"
    if not settings.llm.api_key:
        check("LLM_API_KEY set in .env", False, "add your Groq key to .env")
        return
    headers = {"Authorization": f"Bearer {settings.llm.api_key}"}
    model = settings.llm.model
    with httpx.Client(timeout=30) as client:
        if not model:
            models = client.get(f"{base}models", headers=headers)
            if models.status_code == 200:
                names = [m["id"] for m in models.json().get("data", [])]
                model = next((n for n in FALLBACK_MODELS if n in names), names[0] if names else "")
            else:
                check("LLM /models reachable", False, f"HTTP {models.status_code}: {models.text[:200]}")
                return
        resp = client.post(
            f"{base}chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
                "max_tokens": 500,
            },
        )
        if resp.status_code != 200:
            check("LLM chat completion", False, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return
        reply = resp.json()["choices"][0]["message"]["content"]
        check("LLM chat completion", "PONG" in reply.upper(), f"model={model}, reply={reply!r}")


def check_embeddings() -> None:
    try:
        model = TextEmbedding(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
        vecs = list(model.embed(["Redis vector search for RAG"]))
        dims = len(vecs[0])
        expected = int(os.getenv("settings.embed.dim", "384"))
        check(
            "fastembed produces vector",
            dims == expected,
            f"dims={dims} (expected {expected}, matches FT.CREATE DIM)",
        )
    except Exception as exc:
        check("fastembed", False, str(exc)[:200])


if __name__ == "__main__":
    check_redis()
    check_llm()
    check_embeddings()
    print(f"\n{'-' * 40}\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)