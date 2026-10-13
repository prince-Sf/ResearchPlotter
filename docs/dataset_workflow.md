# Dataset workflow

CSV, clearly delimited TXT tables, Excel, JSON, and Parquet files are loaded into pandas and profiled for types, missing/duplicate rows, constants, low cardinality, ranges, outliers, correlations, and possible time columns. TXT input is accepted only when a recognizable delimiter is present or a consistent multi-column whitespace table with numeric data is detected; ordinary prose is rejected and should use the Formula Plotter instead. Recommendations are structure-driven. If no time-like column exists, the profiler explicitly recommends relationship/distribution plots rather than inventing a time series.

The Noise / Uncertainty view uses the existing seeded Gaussian perturbation
service on a copy of a selected numeric column. It stores clean source values
separately from synthetic observations and labels the run exploratory.
