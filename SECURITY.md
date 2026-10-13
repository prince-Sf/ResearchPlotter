# Security policy

## Reporting a concern

Please do not publish credentials, personal information, or a working exploit
in an issue. Contact the project maintainer privately with a concise
description, affected version or commit, reproduction steps, and impact. No
separate security contact has been configured in this repository yet.

## Safe handling

ResearchPlotter accepts user-provided datasets, formulas, PDFs, and images.
Treat them as untrusted. Mathematical input is parsed through supported,
allowlisted constructs; uploaded text and extracted equations must never be
executed as Python. Do not include secrets, private datasets, or copyrighted
papers in bug reports unless sharing is authorized.

The repository has no selected license and no configured vulnerability
disclosure service. Coordinate security fixes with the maintainer before
public disclosure.
