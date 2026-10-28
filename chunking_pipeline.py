
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path

from paths import CHUNKS_DIR, CLEAN_TXT

# ==========================================================================
# CONFIGURATION
# ==========================================================================
DEFAULT_FILE = CLEAN_TXT    # produced by prepare_dataset.py

FIXED_SIZE = 300            # Approach 1: characters per chunk

WINDOW_WORDS = 45           # Approach 2: ~300 characters of text
STRIDE_WORDS = 37           #   -> 8 words (~50 chars) of overlap per step

SEMANTIC_BUFFER = 1         # Approach 3: compare each unit WITH its neighbours
SEMANTIC_PERCENTILE = 80    #   break at the top 20% biggest topic shifts
SEMANTIC_MIN_CHARS = 80     #   merge fragments smaller than this
SEMANTIC_MAX_CHARS = 900    #   never let a chunk grow beyond this

COLUMN_WIDTH = 38           # width of each column in the 3-way comparison


# ==========================================================================
# PART 1 - PARSING  (markdown / text / PDF  ->  one clean string)
# ==========================================================================
@dataclass
class ParsedDoc:
    text: str
    source: str
    file_type: str
    page_spans: list = field(default_factory=list)  # [(start, end, page_no)]


def load_text_file(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return ParsedDoc(text=text, source=path.name, file_type=path.suffix.lstrip("."))


def load_pdf(path: Path) -> ParsedDoc:
    """Extract text page by page with pypdf and remember where each page
    starts/ends, so every chunk can cite the page(s) it came from."""
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("PDF input needs pypdf:  pip install pypdf")

    reader = PdfReader(str(path))
    parts, spans, offset = [], [], 0
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").replace("\r\n", "\n")
        page_text = re.sub(r"[ \t]+\n", "\n", page_text).strip()
        spans.append((offset, offset + len(page_text), page_no))
        parts.append(page_text)
        offset += len(page_text) + 2          # +2 for the "\n\n" joiner
    return ParsedDoc(text="\n\n".join(parts), source=path.name,
                     file_type="pdf", page_spans=spans)


def load_document(path: Path) -> ParsedDoc:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text_file(path)


# ==========================================================================
# HEADER INDEX  (used to attach section metadata to ANY chunk)
# ==========================================================================
MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# PDFs lose the '#' marks, so we detect numbered headings like "2.1 Hardware Allowance".
# Lines ending in '.' or longer than 70 chars are treated as list items, not headings.
NUM_HEADER_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{2,68}[^.\n])\s*$")


def build_header_index(doc: ParsedDoc):
    """Return [(char_offset, level, title)] for every real heading.
    Ignores '#' lines inside ``` code fences and inside > blockquotes."""
    index, offset, in_code = [], 0, False
    for line in doc.text.split("\n"):
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
        elif not in_code:
            m = MD_HEADER_RE.match(line)
            if m:
                index.append((offset, len(m.group(1)), m.group(2).strip()))
            elif doc.file_type == "pdf":
                n = NUM_HEADER_RE.match(line.strip())
                if n:
                    level = n.group(1).count(".") + 2     # "2." -> h2, "2.1" -> h3
                    index.append((offset, level, line.strip()))
        offset += len(line) + 1
    return index


def header_path_at(offset: int, header_index) -> dict:
    """Which h1/h2/h3/h4 headings are 'in force' at a character offset?"""
    path = {}
    for pos, level, title in header_index:
        if pos > offset:
            break
        path = {k: v for k, v in path.items() if int(k[1]) < level}
        if level <= 4:
            path[f"h{level}"] = title
    return dict(sorted(path.items()))


def section_span(title: str, header_index, text_len: int):
    """(start, end) of a section: from its heading to the next heading of
    the same or higher level."""
    for i, (pos, level, t) in enumerate(header_index):
        if title in t:
            for pos2, level2, _ in header_index[i + 1:]:
                if level2 <= level:
                    return pos, pos2
            return pos, text_len
    return None


# ==========================================================================
# PART 2 - THE THREE CHUNKING STRATEGIES
# Each returns a list of (start_char, end_char[, extra_metadata]) spans.
# Working with spans (not copies of text) keeps citations exact.
# ==========================================================================
def trim_span(text: str, s: int, e: int):
    """Shrink a span so it does not start/end with whitespace."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


# ---------- Approach 1: Fixed-length --------------------------------------
def fixed_length_spans(text: str, size: int = FIXED_SIZE):
    """Cut every `size` characters. No overlap, no awareness of words,
    sentences, lists or tables. The simplest possible baseline."""
    spans = []
    for start in range(0, len(text), size):
        s, e = trim_span(text, start, min(start + size, len(text)))
        if e > s:
            spans.append((s, e, {}))
    return spans


# ---------- Approach 2: Sliding window -----------------------------------
def sliding_window_spans(text: str, window: int = WINDOW_WORDS,
                         stride: int = STRIDE_WORDS):
    """Move a window of `window` words forward `stride` words at a time.
    Consecutive chunks share (window - stride) words. Works on whole words,
    so it never cuts a word in half, and keeps original newlines/tables."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    spans, i = [], 0
    while i < len(words):
        j = min(i + window, len(words))
        spans.append((words[i][0], words[j - 1][1],
                      {"window_words": j - i, "first_word_index": i}))
        if j == len(words):
            break
        i += stride
    return spans


# ---------- Approach 3: Semantic (lexical, embedding-free) ----------------
STOPWORDS = set("""
a an and are as at be been but by can for from has have if in into is it its
may must no not of on or our per such than that the their them then there these
this to up was we were which who will with within without you your any all also
""".split())

LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(*\"'])")


def segment_units(text: str):
    """Break the document into small 'meaning units': sentences, single list
    items, whole tables, whole code blocks and headings. Tables and code are
    kept whole because cutting them destroys their meaning."""
    units = []                       # (start, end, kind)
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line), line))
        pos += len(line) + 1

    i, para = 0, []

    def flush_para():
        if not para:
            return
        ps, pe = para[0][0], para[-1][1]
        cursor = ps
        for m in SENTENCE_END_RE.finditer(text, ps, pe):
            s, e = trim_span(text, cursor, m.start())
            if e > s:
                units.append((s, e, "sentence"))
            cursor = m.end()
        s, e = trim_span(text, cursor, pe)
        if e > s:
            units.append((s, e, "sentence"))
        para.clear()

    while i < len(lines):
        s, e, line = lines[i]
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):            # code block
            flush_para()
            j = i + 1
            while j < len(lines) and not lines[j][2].strip().startswith(("```", "~~~")):
                j += 1
            j = min(j, len(lines) - 1)
            units.append((s, lines[j][1], "code"))
            i = j + 1
            continue
        if stripped.startswith("|"):                       # table
            flush_para()
            j = i
            while j + 1 < len(lines) and lines[j + 1][2].strip().startswith("|"):
                j += 1
            units.append((s, lines[j][1], "table"))
            i = j + 1
            continue
        if not stripped:
            flush_para()
        elif MD_HEADER_RE.match(line):
            flush_para()
            units.append((s, e, "heading"))
        elif LIST_RE.match(line):
            flush_para()
            units.append(trim_span(text, s, e) + ("list_item",))
        else:
            para.append((s, e))
        i += 1
    flush_para()

    # A heading is a label, not a thought: glue it onto the unit after it,
    # so a topic break can never fall between a heading and its content.
    glued, pending = [], None
    for s, e, kind in units:
        if kind == "heading":
            pending = pending if pending is not None else s
            continue
        glued.append((pending if pending is not None else s, e, kind))
        pending = None
    if pending is not None:
        glued.append((pending, units[-1][1], "heading"))

    # Safety valve: a unit bigger than the max (e.g. a PDF table that came out
    # as one long blob of text) is packed into line-, then word-sized pieces.
    final = []
    for s, e, kind in glued:
        if e - s <= SEMANTIC_MAX_CHARS or kind in {"table", "code"}:
            final.append((s, e, kind))
            continue
        pieces = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text[s:e])]
        pieces = [(s + a, s + b) for a, b in pieces]
        cur_s = cur_e = None
        for a, b in pieces:
            if b - a > SEMANTIC_MAX_CHARS:            # one huge line -> words
                for w in re.finditer(r"\S+", text[a:b]):
                    wa, wb = a + w.start(), a + w.end()
                    if cur_s is not None and wb - cur_s > SEMANTIC_MAX_CHARS // 2:
                        final.append((cur_s, cur_e, kind + "_part"))
                        cur_s = None
                    cur_s = wa if cur_s is None else cur_s
                    cur_e = wb
                continue
            if cur_s is not None and b - cur_s > SEMANTIC_MAX_CHARS // 2:
                final.append((cur_s, cur_e, kind + "_part"))
                cur_s = None
            cur_s = a if cur_s is None else cur_s
            cur_e = b
        if cur_s is not None:
            final.append((cur_s, cur_e, kind + "_part"))
    return final


def tokenize(t: str):
    toks = re.findall(r"[a-z0-9]+", t.lower())
    out = []
    for w in toks:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s"):   # tiny stemmer: "rules" -> "rule"
            w = w[:-1]
        out.append(w)
    return out


def cosine(a: dict, b: dict) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def percentile(values, p):
    if not values:
        return 1.0
    v = sorted(values)
    return v[int(round(p / 100 * (len(v) - 1)))]


def semantic_spans(text: str):
    """Embedding-free semantic chunking:
    1. split into meaning units                      (segment_units)
    2. turn each unit + its neighbours into a TF-IDF word vector
    3. distance between neighbours = 1 - cosine similarity
    4. cut where distance is in the top (100 - SEMANTIC_PERCENTILE)%
    5. enforce max size (split at biggest internal shift) and min size (merge)"""
    units = segment_units(text)
    if len(units) < 2:
        return [(units[0][0], units[0][1], {})] if units else []

    tfs = [Counter(tokenize(text[s:e])) for s, e, _ in units]
    df = Counter(w for tf in tfs for w in tf)
    n = len(units)
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def window_vec(i):
        combined = Counter()
        for k in range(max(0, i - SEMANTIC_BUFFER), min(n, i + SEMANTIC_BUFFER + 1)):
            combined.update(tfs[k])
        return {w: c * idf[w] for w, c in combined.items()}

    vecs = [window_vec(i) for i in range(n)]
    dist = [1 - cosine(vecs[i], vecs[i + 1]) for i in range(n - 1)]  # gap after unit i
    threshold = percentile(dist, SEMANTIC_PERCENTILE)

    # step 4: initial groups
    groups, cur = [], [0]
    for i in range(n - 1):
        if dist[i] > threshold:
            groups.append(cur)
            cur = []
        cur.append(i + 1)
    groups.append(cur)

    size = lambda g: units[g[-1]][1] - units[g[0]][0]

    # step 5a: split oversized groups at their largest internal topic shift
    def split_big(g):
        if size(g) <= SEMANTIC_MAX_CHARS or len(g) == 1:
            return [g]
        k = max(range(len(g) - 1), key=lambda x: dist[g[x]])
        return split_big(g[:k + 1]) + split_big(g[k + 1:])
    groups = [part for g in groups for part in split_big(g)]

    # step 5b: merge tiny fragments (e.g. a lone heading) into the next group
    merged, i = [], 0
    while i < len(groups):
        g = groups[i]
        while (size(g) < SEMANTIC_MIN_CHARS and i + 1 < len(groups)
               and size(g + groups[i + 1]) <= SEMANTIC_MAX_CHARS):
            i += 1
            g = g + groups[i]
        merged.append(g)
        i += 1

    spans = []
    for g in merged:
        last = g[-1]
        spans.append((units[g[0]][0], units[last][1], {
            "num_units": len(g),
            "unit_kinds": sorted({units[k][2] for k in g}),
            "boundary_distance": round(dist[last], 3) if last < n - 1 else None,
            "break_threshold": round(threshold, 3),
        }))
    return spans


# ==========================================================================
# PART 3 - METADATA
# ==========================================================================
def detect_content_types(t: str) -> list:
    lines = t.splitlines()
    types = []
    if any(MD_HEADER_RE.match(l) for l in lines):
        types.append("heading")
    if any(re.match(r"^\s*[-*+]\s+", l) for l in lines):
        types.append("bullet_list")
    if any(re.match(r"^\s*\d+\.\s+", l) for l in lines):
        types.append("numbered_list")
    if any(l.strip().startswith("|") and l.count("|") >= 3 for l in lines):
        types.append("table")
    if "```" in t:
        types.append("code")
    if any(l.lstrip().startswith(">") for l in lines):
        types.append("blockquote")
    if not types or any(len(l) > 120 and not l.lstrip().startswith(("|", "-")) for l in lines):
        types.append("paragraph")
    return types


def has_orphan_heading(t: str) -> bool:
    """Chunk ENDS with a heading -> its content landed in another chunk."""
    lines = [l for l in t.splitlines() if l.strip()]
    return bool(lines) and bool(MD_HEADER_RE.match(lines[-1]))


def table_is_broken(t: str) -> bool:
    """Has table rows but not the |---| separator row => the column names
    were cut off, or a row was sliced in half."""
    rows = [l.strip() for l in t.splitlines()
            if l.count("|") >= 3 and (l.strip().startswith("|") or l.strip().endswith("|"))]
    if not rows:
        return False
    has_sep = any(re.fullmatch(r"\|[\s\-:|]+\|", r) for r in rows)
    sliced_row = any(not (r.startswith("|") and r.endswith("|")) for r in rows)
    return (not has_sep) or sliced_row


def pages_for(span, page_spans):
    s, e = span
    return [p for ps, pe, p in page_spans if s < pe and e > ps]


def build_chunks(doc: ParsedDoc, strategy: str, spans, header_index):
    text, chunks = doc.text, []
    for i, (s, e, extra) in enumerate(spans):
        body = text[s:e]
        # look up the section at the first line of real content, so a chunk
        # starting "## 1. ...\n\n### 1.1 ..." is labelled with BOTH headings
        content_at = s
        for m in re.finditer(r"[^\n]*\n?", body):
            if m.group().strip() and not MD_HEADER_RE.match(m.group().strip()):
                content_at = s + m.start()
                break
        path = header_path_at(content_at, header_index)
        line_start = text.count("\n", 0, s) + 1
        meta = {
            # identity & provenance (citations)
            "chunk_id": f"{Path(doc.source).stem}::{strategy}::{i:03d}::"
                        f"{hashlib.sha1(body.encode()).hexdigest()[:8]}",
            "source": doc.source,
            "file_type": doc.file_type,
            "strategy": strategy,
            "chunk_index": i,
            "start_char": s,
            "end_char": e,
            "line_start": line_start,
            "line_end": line_start + body.count("\n"),
            # size
            "char_count": len(body),
            "word_count": len(body.split()),
            # structure (filtering & readable citations)
            "section_path": " > ".join(path.values()),
            **path,
            "content_types": detect_content_types(body),
            # quality flags (for evaluation / QA)
            "orphan_heading": has_orphan_heading(body),
            "broken_table": table_is_broken(body),
            "cut_mid_word": (s > 0 and text[s - 1].isalnum() and text[s].isalnum())
                            or (e < len(text) and text[e - 1].isalnum() and text[e].isalnum()),
            **extra,
        }
        if doc.page_spans:
            meta["pages"] = pages_for((s, e), doc.page_spans)
        chunks.append({"text": body, "metadata": meta})

    # overlap with previous chunk, measured exactly from spans
    for prev, cur in zip(chunks, chunks[1:]):
        cur["metadata"]["overlap_with_prev_chars"] = max(
            0, prev["metadata"]["end_char"] - cur["metadata"]["start_char"])
    if chunks:
        chunks[0]["metadata"]["overlap_with_prev_chars"] = 0
    return chunks


# ==========================================================================
# PART 4 - COMPARISON & REPORTING
# ==========================================================================
def summarize(chunks) -> dict:
    m = [c["metadata"] for c in chunks]
    sizes = [x["char_count"] for x in m]
    dup = sum(x["overlap_with_prev_chars"] for x in m)
    return {
        "chunks": len(m),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_chars": round(sum(sizes) / len(sizes)),
        "orphan_headings": sum(x["orphan_heading"] for x in m),
        "broken_table_chunks": sum(x["broken_table"] for x in m),
        "chunks_cut_mid_word": sum(x["cut_mid_word"] for x in m),
        "duplicated_chars": dup,
        "duplication_pct": round(100 * dup / sum(sizes), 1),
    }


def print_summary_table(results: dict):
    names = list(results)
    stats = {n: summarize(results[n]) for n in names}
    keys = list(next(iter(stats.values())))
    w = 22
    print(f"{'metric':<{w}}" + "".join(f"{n:>18}" for n in names))
    print("-" * (w + 18 * len(names)))
    for k in keys:
        print(f"{k:<{w}}" + "".join(f"{str(stats[n][k]):>18}" for n in names))


def wrap_block(t: str, width: int):
    out = []
    for line in t.splitlines() or [""]:
        out.extend(textwrap.wrap(line, width, drop_whitespace=False) or [""])
    return out


def three_way(columns: dict):
    """columns = {title: [chunk, ...]} -> printed side by side."""
    rendered = []
    for title, chunks in columns.items():
        lines = [f"{title} ({len(chunks)} chunks)", "=" * COLUMN_WIDTH]
        for c in chunks:
            m = c["metadata"]
            flags = [f for f, on in (("ORPHAN-HEAD", m["orphan_heading"]),
                                     ("BROKEN-TABLE", m["broken_table"]),
                                     ("MID-WORD", m["cut_mid_word"])) if on]
            lines.append(f"#{m['chunk_index']} {m['char_count']}ch "
                         f"ov={m['overlap_with_prev_chars']} {' '.join(flags)}")
            lines += wrap_block(c["text"], COLUMN_WIDTH)
            lines.append("-" * COLUMN_WIDTH)
        rendered.append(lines)
    for row in zip_longest(*rendered, fillvalue=""):
        print(" | ".join(f"{cell[:COLUMN_WIDTH]:<{COLUMN_WIDTH}}" for cell in row))


def compare_section(title, results, header_index, text_len):
    span = section_span(title, header_index, text_len)
    if not span:
        print(f"\n(section '{title}' not found)")
        return
    s, e = span
    print("\n" + "#" * (COLUMN_WIDTH * 3 + 6))
    print(f"SAME SOURCE SECTION: '{title}'  (chars {s}-{e})")
    print("#" * (COLUMN_WIDTH * 3 + 6))
    three_way({
        name: [c for c in chunks
               if c["metadata"]["start_char"] < e and c["metadata"]["end_char"] > s]
        for name, chunks in results.items()
    })


# ==========================================================================
# MAIN
# ==========================================================================
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0] if argv else DEFAULT_FILE)
    if not path.exists():
        sys.exit(f"File not found: {path}\nRun prepare_dataset.py (or: python run_rag.py prepare) first.")

    doc = load_document(path)                       # PART 1: parse
    header_index = build_header_index(doc)
    print(f"Source: {doc.source} | type: {doc.file_type} | {len(doc.text)} chars | "
          f"{len(header_index)} headings detected"
          + (f" | {len(doc.page_spans)} pages" if doc.page_spans else ""))

    results = {                                     # PART 2 + 3: chunk + metadata
        "fixed_length": build_chunks(doc, "fixed_length",
                                     fixed_length_spans(doc.text), header_index),
        "sliding_window": build_chunks(doc, "sliding_window",
                                       sliding_window_spans(doc.text), header_index),
        "semantic": build_chunks(doc, "semantic",
                                 semantic_spans(doc.text), header_index),
    }

    print("\n" + "=" * 30, "SUMMARY", "=" * 30)      # PART 4: compare
    print_summary_table(results)

    print("\n" + "=" * 20, "ORPHANED HEADINGS (chunk ends with a heading)", "=" * 20)
    for name, chunks in results.items():
        bad = [c for c in chunks if c["metadata"]["orphan_heading"]]
        print(f"{name}: {len(bad)}" + "".join(
            f"\n   #{c['metadata']['chunk_index']} ends with "
            f"{c['text'].splitlines()[-1]!r}" for c in bad[:4]))

    print("\n" + "=" * 20, "WHAT THE SLIDING-WINDOW OVERLAP DUPLICATED", "=" * 20)
    sw = results["sliding_window"]
    for c in sw[1:5]:
        m = c["metadata"]
        print(f"#{m['chunk_index']} repeats {m['overlap_with_prev_chars']} chars: "
              f"{c['text'][:m['overlap_with_prev_chars']]!r}")

    for section in ("Hardware Allowance", "Remote Work Rules", "Eligibility and Philosophy"):
        compare_section(section, results, header_index, len(doc.text))

    sample = results["semantic"][min(5, len(results["semantic"]) - 1)]["metadata"]
    print("\nSample metadata (semantic chunk):")
    print(json.dumps(sample, indent=2))

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chunks in results.items():            # save for export_reports.py
        out = CHUNKS_DIR / f"chunks_{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Saved {len(chunks):>3} chunks -> {out.name}")


if __name__ == "__main__":
    main()
