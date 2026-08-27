\# Project: ML Dataset Profiler (working name)



\## One-liner

A Windows-installable, no-code desktop app that opens AI-training-dataset

files (parquet, arrow, csv) and generates ML-specific profiling visuals —

null patterns, class balance, embedding projections, image/audio previews —

plus an LLM-driven "ask for a custom chart" feature. One-time purchase,

on-prem by default.



\## Target user

ML/data engineers and data annotation teams who need to sanity-check a

training dataset without writing pandas code every time.



\## Tech stack (decided — do not change without discussion)

\- Language: Python 3.11+

\- Data layer: polars (primary), pyarrow for arrow/parquet specifics,

&#x20; duckdb available for ad-hoc SQL later

\- UI shell: pywebview (native window wrapping a local HTML/JS frontend)

\- Frontend: plain HTML/CSS/JS + Plotly.js for charts (no framework yet —

&#x20; keep it simple until the app has real features)

\- Packaging: PyInstaller -> Windows .exe (do NOT set this up yet — Phase 5)



\## Current phase: PHASE 2

Build exactly this and stop:

\- Detect embedding columns: any column typed as a list of floats (fixed-size or variable-length) with a consistent length across rows, greater than 8 dimensions

\- For each detected embedding column, project to 2D via PCA (scikit-learn) and show it as a Plotly scatter plot

\- If the dataset has a categorical column, let the user pick one to color the scatter points by (unpicked = single color)

\- For datasets over 10,000 rows, randomly sample down to 10,000 rows before running PCA and plotting, and show a note in the UI that it's a sample

\- UMAP is not part of this pass — PCA only, per the roadmap's "PCA then UMAP" note; UMAP stays open for a later pass if needed



\## Explicit non-goals right now

\- Do not touch the LLM custom-chart feature yet (that's Phase 4)

\- Do not set up PyInstaller/installer yet (that's Phase 5)

\- Do not add authentication or license-key checks yet



\## Conventions

\- Type hints throughout

\- Keep data-loading logic separate from UI code, so it's testable headless

&#x20; (i.e. importable and callable without launching the pywebview window)

\- Small, incremental commits — commit as soon as something works, before

&#x20; starting the next small piece

\- Prefer polars over pandas for anything performance-sensitive



\## Roadmap (for context only — do not build ahead of current phase)

1\. Phase 0: open file, show schema + table (this brief's scope)

2\. Phase 1: null heatmap, numeric histograms, class balance, text-length

&#x20;  distributions

3\. Phase 2: embedding column detection, 2D projection (PCA then UMAP),

&#x20;  scatter plot colored by label — this is the core differentiator

4\. Phase 3: inline preview for image/audio/text columns

5\. Phase 4: natural-language custom chart via LLM (declarative chart-spec

&#x20;  first; sandboxed code-gen only as a fallback for non-standard requests)

6\. Phase 5: packaging, installer, licensing

