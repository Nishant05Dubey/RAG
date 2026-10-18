"""Day 06 retrieval engine over the Day 05 chunk corpus.

Pure Python + NumPy: no vector database, no LLM, no agents. The same
SentenceTransformer model embeds chunks and queries. This module knows
nothing about test cases; see run_queries.py for execution and logging.

    from retriever import Retriever
    r = Retriever(index="child")
    r.retrieve("<your question>", top_k=3)
"""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from paths import DOC_METADATA, RETRIEVAL_CHUNKS, require

CHUNKS_PATH = RETRIEVAL_CHUNKS
DOC_METADATA_PATH = DOC_METADATA

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE v1.5 documents this instruction for short query -> passage retrieval.
# It is applied to queries only; passages are embedded as-is.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

INDEX_MODES = ("parent", "child")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.M)
PAGE_RE = re.compile(r"\[\[PAGE (\d+)\]\]")
# Day 05 chunker diagnostics that describe the whole parent chunk; children
# drop them (they stay reachable through parent_chunk_id).
PARENT_ONLY_FIELDS = {
    "content_types", "orphan_heading", "broken_table", "cut_mid_word", "num_units",
    "unit_kinds", "boundary_distance", "break_threshold", "overlap_with_prev_chars",
}

_models = {}


def get_model(name=MODEL_NAME):
    if name not in _models:
        _models[name] = SentenceTransformer(name, device="cpu")
    return _models[name]


def load_chunks(path=CHUNKS_PATH):
    """Day 05 chunks: [{chunk_id, text, metadata}, ...]."""
    require(Path(path), "Generate the chunks first: python run_rag.py prepare")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_document_metadata(path=DOC_METADATA_PATH):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --------------------------------------------------------------------------
# Heading-aligned child chunks (Day 06 experiment, Option C)
# --------------------------------------------------------------------------
def _split_points(text):
    """Offsets where a new section starts inside a chunk.

    A split happens before a heading line, except when the text before it
    is only headings (e.g. '## 3. Leave' directly followed by '### 3.1'),
    so a heading always stays attached to the content beneath it.
    """
    points = []
    for m in HEADING_RE.finditer(text):
        start = m.start()
        prev = text[points[-1] if points else 0:start]
        if start > 0 and HEADING_RE.sub("", prev).strip():
            points.append(start)
    return points


def split_at_headings(chunks):
    """Split each Day 05 chunk at the headings inside it.

    Children keep every parent metadata field plus `parent_chunk_id`, and
    get a `section_path` rebuilt from a document-order heading stack, so a
    child's label is the section it actually contains (Day 05 labels a
    chunk only by the section where it starts).
    """
    stack = {}
    children = []
    for parent in chunks:
        text = parent["text"]
        pm = {k: v for k, v in parent["metadata"].items()
              if k not in PARENT_ONLY_FIELDS and not re.fullmatch(r"h[1-4]", k)}
        cuts = [0] + _split_points(text) + [len(text)]
        for n, (a, b) in enumerate(zip(cuts, cuts[1:])):
            raw = text[a:b]
            piece = raw.strip("\n")
            if not piece.strip():
                continue
            a += len(raw) - len(raw.lstrip("\n"))  # offset of piece in parent
            for m in HEADING_RE.finditer(piece):
                level = len(m.group(1))
                stack = {k: v for k, v in stack.items() if k < level}
                stack[level] = m.group(2)
            if not stack and pm["section_path"]:
                stack = dict(enumerate(pm["section_path"].split(" > "), start=1))
            before = [int(p) for p in PAGE_RE.findall(text[:a])]
            first_page = before[-1] if before else pm["pages"][0]
            inside = [int(p) for p in PAGE_RE.findall(piece)]
            line_start = pm["line_start"] + text[:a].count("\n")
            meta = {
                **pm,
                "chunk_id": f"{parent['chunk_id']}::c{n}",
                "parent_chunk_id": parent["chunk_id"],
                "child_index": n,
                "start_char": pm["start_char"] + a,
                "end_char": pm["start_char"] + a + len(piece),
                "line_start": line_start,
                "line_end": line_start + piece.count("\n"),
                "char_count": len(piece),
                "word_count": len(piece.split()),
                "section_path": " > ".join(stack[k] for k in sorted(stack)),
                **{f"h{k}": v for k, v in sorted(stack.items())},
                "pages": sorted({first_page, *inside}),
            }
            children.append({"chunk_id": meta["chunk_id"], "text": piece, "metadata": meta})
    for i, c in enumerate(children):
        c["metadata"]["chunk_index"] = i
    return children


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------
class Retriever:
    """In-memory cosine-similarity retriever over Day 05 chunks.

    index="parent": the Day 05 chunks as-is (Day 06 baseline).
    index="child":  heading-aligned children of those chunks (experiment).
    """

    def __init__(self, index="parent", chunks_path=CHUNKS_PATH, model_name=MODEL_NAME):
        if index not in INDEX_MODES:
            raise ValueError(f"index must be one of {INDEX_MODES}, got {index!r}")
        self.index = index
        self.model_name = model_name
        self.document_metadata = load_document_metadata()
        parents = load_chunks(chunks_path)
        self.chunks = parents if index == "parent" else split_at_headings(parents)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.matrix = self._embed([c["text"] for c in self.chunks])  # (n, dim)

    def _embed(self, texts):
        """L2-normalised embeddings, so dot product == cosine similarity."""
        vecs = get_model(self.model_name).encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def similarities(self, query):
        """Cosine similarity of the query against every chunk, index order."""
        return self.matrix @ self._embed([QUERY_INSTRUCTION + query])[0]

    def rank_all(self, query):
        """Full ranking over the whole index: [(position, cosine), ...]."""
        s = self.similarities(query)
        return [(int(i), float(s[i])) for i in np.argsort(-s)]

    def source(self, chunk):
        m = chunk["metadata"]
        return {
            "document": m.get("original_source", m["source"]),
            "document_title": m.get("document_title"),
            "pages": m["pages"],
            "section": m["section_path"].split(" > ")[-1] if m["section_path"] else "",
            "section_path": m["section_path"],
            "parent_chunk_id": m.get("parent_chunk_id"),
        }

    def format_hit(self, rank, position, score):
        chunk = self.chunks[position]
        return {
            "rank": rank,
            "similarity_score": round(score, 4),
            "chunk_id": chunk["chunk_id"],
            "source": self.source(chunk),
            "chunk_text": chunk["text"],
        }

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        return [
            self.format_hit(r, pos, score)
            for r, (pos, score) in enumerate(self.rank_all(query)[:top_k], start=1)
        ]

    def find_chunk(self, fragment):
        """Chunks whose id contains `fragment` (for raw-chunk inspection)."""
        return [c for c in self.chunks if fragment in c["chunk_id"]]

    def describe(self):
        return {
            "model": self.model_name,
            "metric": "cosine",
            "index": self.index,
            "num_chunks": len(self.chunks),
            "chunks_path": str(CHUNKS_PATH),
            "document": self.document_metadata.get("document_title"),
            "document_id": self.document_metadata.get("document_id"),
        }
