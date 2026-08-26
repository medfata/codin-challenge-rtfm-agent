"""Fast int8 embedding for BAAI/bge-small-en-v1.5 via ONNX Runtime.

Uses the well-formed int8 export from Xenova/bge-small-en-v1.5 (same base
model as fastembed's BAAI/bge-small-en-v1.5) and replicates fastembed/BGE
inference: CLS pooling + L2 normalization. ~7x faster than fp32 on CPU.
"""

import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from rtfm_agent.config import settings

REPO_ID = "Xenova/bge-small-en-v1.5"
MAX_SEQ_LEN = 512


class FastInt8Embedder:
    """Embed texts with bge-small-en-v1.5 (int8, CLS pooling, L2-normalized)."""

    def __init__(self, threads: int = 2):
        model_path = hf_hub_download(REPO_ID, "onnx/model_int8.onnx")
        tokenizer_path = hf_hub_download(REPO_ID, "tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        self.tokenizer.enable_padding()  # pads to longest in batch

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return (n, dim) float32 array of embeddings, in input order."""
        # Sort by length so each batch pads to a similar length
        # (huge speedup when text lengths vary), then restore order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]

        out = np.empty((len(texts), self.session.get_outputs()[0].shape[-1]), dtype=np.float32)
        for start in range(0, len(sorted_texts), batch_size):
            batch_texts = sorted_texts[start:start + batch_size]
            enc = self.tokenizer.encode_batch(batch_texts)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            token_type = np.zeros_like(ids)

            hidden = self.session.run(
                None,
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": token_type,
                },
            )[0]

            # CLS pooling + L2 normalize (BGE recipe)
            cls_vecs = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(cls_vecs, axis=1, keepdims=True)
            cls_vecs = cls_vecs / np.maximum(norms, 1e-12)

            batch_indices = order[start:start + batch_size]
            out[batch_indices] = cls_vecs

        return out


_embedder: FastInt8Embedder | None = None


def get_embedder() -> FastInt8Embedder:
    """Process-wide singleton; first call loads the ONNX model."""
    global _embedder
    if _embedder is None:
        _embedder = FastInt8Embedder(threads=settings.embed.ort_threads)
    return _embedder


def embed_question(text: str) -> bytes:
    """float32 embedding blob for one text - the wire format for KNN PARAMS."""
    vec = get_embedder().embed([text])[0].astype(np.float32)
    return vec.tobytes()


if __name__ == "__main__":
    import time

    t0 = time.time()
    emb = FastInt8Embedder(threads=int(os.getenv("ORT_THREADS", "2")))
    print(f"model loaded in {time.time() - t0:.1f}s")

    texts = ["hello world", "git is a distributed version control system"] * 8
    t1 = time.time()
    vecs = emb.embed(texts)
    dt = time.time() - t1
    print(f"embedded {len(vecs)} texts in {dt:.2f}s ({len(vecs)/dt:.0f} texts/s)")
    print("shape:", vecs.shape, "| norm:", np.linalg.norm(vecs[0]))
