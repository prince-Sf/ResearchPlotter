from __future__ import annotations

import re

from research_plotter.ingestion.pdf_loader import extract_pdf_pages


def extract_table_mentions(path: str) -> list[dict[str, object]]:
    tables = []
    for page_number, page in enumerate(extract_pdf_pages(path), 1):
        for match in re.finditer(r"(?im)^\s*table\s+(\d+)\.?\s*(.*)$", page):
            tables.append({"table_id": f"table_{match.group(1)}", "caption": match.group(2).strip(), "source_page": page_number, "status": "PARTIAL"})
    return tables
