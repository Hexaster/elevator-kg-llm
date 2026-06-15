# Knowledge Graph-Enhanced LLMs for Reliable Elevator Fault Diagnosis

Source code for the elevator-maintenance knowledge-graph (KG) construction and
knowledge-enhanced question-answering pipeline described in the paper
*"Knowledge Graph-Enhanced Large Language Models for Reliable Elevator Fault
Diagnosis"*.

## Pipeline

The full pipeline is run by `run.sh`, which executes the stages in order:

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `data_clean.py`      | Clean raw maintenance records |
| 2 | `chunk_split.py`     | Split cleaned records into chunks |
| 3 | `build_kg.py`        | Extract schema-constrained triples (LLM) |
| 4 | `kg_deduplicated.py` | Deduplicate triples/entities |
| 5 | `kg_align.py`        | Embedding-based entity alignment + neuro-symbolic completion |
| 6 | `kgqa.py`            | Retrieve-then-verify KG-augmented question answering |

`utils.py` holds shared I/O and the evaluation functions (Key Entity Recall,
BERTScore, embedding cosine). `r1_3_stats.py` reproduces the per-item Key Entity
Recall statistics reported in the paper (sample size, paired Wilcoxon signed-rank
test, and bootstrap 95% confidence intervals).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your own keys / paths
```

Configuration is read from environment variables (see `.env.example`):

- `LLM_API_KEY`, `LLM_BASE_URL` — the LLM endpoint used for triple extraction,
  retrieve-then-verify link validation, and answer generation.
- `DASHSCOPE_API_KEY` — embedding API for the optional sentence-cosine metric.
- `EMBEDDING_MODEL_PATH` — local path or HuggingFace id of the sentence-embedding
  model used for entity alignment (default `Qwen/Qwen3-Embedding-0.6B`).
- `DATA_DIR` — directory holding the data files (default `./data`).

No API keys are bundled with this repository; supply your own.

## Data availability

The underlying elevator maintenance records and the constructed knowledge graph
are **not** included in this repository owing to proprietary restrictions from
the data provider. They are available from the corresponding author on
reasonable request. The scripts expect these files under `DATA_DIR` (see each
script for the expected filenames).

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper (citation to be updated on
publication).
