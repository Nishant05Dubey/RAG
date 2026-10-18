import hashlib
import json
import re

from pypdf import PdfReader

from paths import CLEAN_TXT, CLEANING_REPORT, DOC_METADATA, PROCESSED_DIR, SOURCE_PDF, require


SOURCE = SOURCE_PDF
OUT = PROCESSED_DIR

# These are verified top-level section titles from this handbook.
TOP_LEVEL_TITLES = {
    "Remote Work Policy",
    "IT Equipment and Security",
    "Leave and Time Off",
    "Information Security and Acceptable Use",
    "Software and SaaS Procurement",
    "Travel and Expenses",
    "Code of Conduct",
    "Performance and Development",
    "Onboarding and Offboarding",
    "Glossary",
    "Contacts",
}

TITLE = "Acme Corp IT and HR Policy Handbook"


def restore_heading(line):
    stripped = line.strip()

    if stripped == TITLE:
        return "# " + stripped

    # Numbered subsections, such as 2.1 and 4.4.1.
    match = re.fullmatch(
        r"(\d+\.\d+(?:\.\d+)?)\s+(.+)", stripped
    )
    if match:
        level = match.group(1).count(".") + 2
        return "#" * level + " " + stripped

    # Restrict H2 recognition to known section titles.
    # This avoids treating numbered procedure steps as headings.
    match = re.fullmatch(r"\d+\.\s+(.+)", stripped)
    if match and match.group(1) in TOP_LEVEL_TITLES:
        return "## " + stripped

    return line


def main():
    require(SOURCE, "Put the policy PDF in data/source/.")
    OUT.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(SOURCE)
    raw_pages = []
    clean_pages = []
    empty_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text(extraction_mode="layout") or ""

        if not extracted.strip():
            empty_pages.append(page_number)

        marker = f"[[PAGE {page_number}]]"
        raw_pages.append(marker + "\n" + extracted)

        cleaned = extracted.replace("\r\n", "\n").replace("\r", "\n")

        # This PDF's bullet glyph may be extracted as DEL.
        # Convert it only when it occurs at the beginning of a line.
        cleaned = re.sub(
            r"(?m)^([ \t]*)\x7f[ \t]*",
            r"\1- ",
            cleaned,
        )

        # Remove nonprinting C0 controls while preserving tabs/newlines.
        cleaned = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned
        )

        cleaned = "\n".join(
            restore_heading(line) for line in cleaned.splitlines()
        )
        clean_pages.append(marker + "\n" + cleaned)

    raw_text = "\n\n".join(raw_pages)
    clean_text = "\n\n".join(clean_pages)

    if not clean_text.strip() or len(empty_pages) == len(reader.pages):
        raise ValueError("No usable PDF text was extracted.")

    (OUT / "extracted_raw.txt").write_text(raw_text, encoding="utf-8")
    CLEAN_TXT.write_text(clean_text, encoding="utf-8")

    metadata = {
        "original_source": SOURCE.name,
        "original_source_sha256": hashlib.sha256(
            SOURCE.read_bytes()
        ).hexdigest(),
        "document_title": TITLE,
        "document_id": "ACME-POL-2026-014",
        "version": "4.0",
        "owner": "People & Technology Office",
        "last_reviewed": "2026-09-01",
        "page_count": len(reader.pages),
    }

    report = {
        "raw_characters": len(raw_text),
        "clean_characters": len(clean_text),
        "empty_pages": empty_pages,
        "replacement_characters": clean_text.count("�"),
        "manual_review_confirmed": False,
        "review_tasks": [
            "Check all six logical tables, including page continuations.",
            "Check nested lists and numbered procedure steps.",
            "Restore code fences and blockquote markers where needed.",
            "Check accented characters, dashes, amounts, and times.",
            "Preserve policy inconsistencies rather than correcting facts.",
        ],
    }

    for path, value in (
        (DOC_METADATA, metadata),
        (CLEANING_REPORT, report),
    ):
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Extracted {len(reader.pages)} pages.")
    print("Saved raw extraction, cleaned TXT, metadata, and cleaning report.")
    print(f"Review {CLEAN_TXT.name} before running chunking.")


if __name__ == "__main__":
    main()
