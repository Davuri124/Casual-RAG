"""Dense retrieval primitives. sentence-transformers is imported lazily so torch-free stages still import."""
import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_id="BAAI/bge-small-en-v1.5", device=None):
        from sentence_transformers import SentenceTransformer
        import torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_id, device=self.dev)
        if self.dev == "cuda":
            self.model.half()
        self.model_id = model_id

    def encode(self, texts, bs=256, query=False, progress=False):
        if query and "bge" in self.model_id:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        e = self.model.encode(list(texts), batch_size=bs, normalize_embeddings=True,
                              show_progress_bar=progress, convert_to_numpy=True)
        return e.astype("float32")


class DenseIndex:
    """Exact cosine search over L2-normalised embeddings (numpy; 3e5 x 384 is tiny)."""

    def __init__(self, emb):
        self.emb = np.ascontiguousarray(emb.astype("float32"))

    def scores(self, q):
        return q @ self.emb.T

    def search(self, q, k, allowed=None):
        """q: (nq,d). Returns (idx (nq,k), sim (nq,k)). `allowed`: optional boolean mask over the index."""
        out_i, out_s = [], []
        for i in range(0, len(q), 64):
            s = self.scores(q[i:i + 64])
            if allowed is not None:
                s[:, ~allowed] = -9
            k_ = min(k, s.shape[1])
            part = np.argpartition(-s, k_ - 1, axis=1)[:, :k_]
            ps = np.take_along_axis(s, part, 1)
            o = np.argsort(-ps, 1)
            out_i.append(np.take_along_axis(part, o, 1)); out_s.append(np.take_along_axis(ps, o, 1))
        return np.vstack(out_i), np.vstack(out_s)
