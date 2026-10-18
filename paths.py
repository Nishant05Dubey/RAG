"""Every file location used by the pipeline, derived from this folder.

Paths are built from __file__, so the scripts work no matter which
directory the terminal is in when you launch them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"          # the original policy PDF
PROCESSED_DIR = DATA_DIR / "processed"    # Day 05 extraction + cleaning
CHUNKS_DIR = DATA_DIR / "chunks"          # Day 05 chunks (read by Day 06)
OUTPUTS_DIR = ROOT / "outputs"            # Day 05 strategy comparison
LOGS_DIR = ROOT / "logs"                  # Day 06 retrieval logs and runs
REPORTS_DIR = ROOT / "reports"            # written evaluation report(s)

SOURCE_PDF = SOURCE_DIR / "Acme_Corp_IT_and_HR_Policy.pdf"
# Keep this filename: chunk ids are built from it ("cleaned_policy::semantic::...").
CLEAN_TXT = PROCESSED_DIR / "cleaned_policy.txt"
DOC_METADATA = PROCESSED_DIR / "document_metadata.json"
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.json"

STRATEGIES = ("fixed_length", "sliding_window", "semantic")
# The chunk file Day 06 retrieves over ({chunk_id, text, metadata} records).
RETRIEVAL_CHUNKS = CHUNKS_DIR / "metadata_semantic.json"
TEST_QUERIES = ROOT / "test_queries.json"


def require(path: Path, hint: str = "") -> Path:
    """Stop with a clear message if a required input file is missing."""
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}" + (f"\n{hint}" if hint else ""))
    return path
