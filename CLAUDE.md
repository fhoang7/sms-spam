# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMS spam detection comparing traditional ML approaches against LLM-based classification. Uses ChromaDB for local vector storage and text embeddings.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f ml.yml
conda activate sms_ml
```

Environment includes: Python 3.10, scikit-learn, optuna, pandas, lazypredict, and ChromaDB (via pip).

## Data Architecture

**Dataset**: `spam.csv` (gitignored)
- Must be read with `encoding='latin'` (latin-1), not UTF-8
- Contains columns: `v1` (label: ham/spam), `v2` (message text)

**Vector Storage**: ChromaDB with local persistence
- Database location: `./data/chroma.sqlite3`
- The `data/` directory is created at runtime if missing
- Uses local file-based storage, not client-server mode

## Code Structure

**embed.py**: Main embedding script
- Loads and preprocesses spam.csv
- Sets up ChromaDB local persistence
- Uses Jupyter-style cell markers (`#%%`) for interactive execution
