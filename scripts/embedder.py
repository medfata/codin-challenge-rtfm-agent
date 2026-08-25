"""Backward-compatible shim; implementation lives in rtfm_agent.embedder."""

from rtfm_agent.embedder import REPO_ID, FastInt8Embedder, MAX_SEQ_LEN

__all__ = ["REPO_ID", "FastInt8Embedder", "MAX_SEQ_LEN"]

if __name__ == "__main__":
    import os
    import time

    t0 = time.time()
    emb = FastInt8Embedder(threads=int(os.getenv("ORT_THREADS", "2")))
    print(f"model loaded in {time.time() - t0:.1f}s")

    texts = ["hello world", "git is a distributed version control system"] * 8
    t1 = time.time()
    vecs = emb.embed(texts)
    dt = time.time() - t1
    print(f"embedded {len(vecs)} texts in {dt:.2f}s ({len(vecs)/dt:.0f} texts/s)")
