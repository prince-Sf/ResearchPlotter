from __future__ import annotations

import re

from research_plotter.ingestion.pdf_loader import extract_pdf_text


def detect_methods(path: str) -> list[str]:
    text = extract_pdf_text(path)
    candidates = re.findall(r"(?i)\b(?:Grünwald[- ]Letnikov|Atangana[- ]Baleanu|Caputo|Mittag[- ]Leffler|solve_ivp|NSFD|PINN|neural network)\b", text)
    return sorted(set(candidates))
