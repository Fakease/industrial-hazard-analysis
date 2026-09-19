# Safety-semantic topic analysis of industrial hazard records

Implementation of multi-LLM structuring, semantic-channel clustering, multistage analysis, ablation and topic-level safety-control synthesis (K1).

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m pip install -e ".[clustering]"
```

Dependency versions are listed in `requirements-lock.txt`.

## Main pipeline

Set `QWEN_API_KEY`, `DEEPSEEK_API_KEY` and `DOUBAO_API_KEY` in the environment, or copy `.env.example` to `.env.local` and supply the values locally.

Prepare an input CSV or XLSX with `raw_record_id` and `raw_text` columns for record identifiers and hazard descriptions:

```bash
python scripts/run_experiment_pipeline.py --config config/experiment.yaml --input /path/to/input.csv --id-column raw_record_id --text-column raw_text --output-root /path/to/local-runs --run-name experiment --stage all
```

| Role | API model ID |
| --- | --- |
| Qwen structuring and topic naming | `qwen-3.7-flash` |
| DeepSeek structuring and context separation | `deepseek-v4-flash` |
| Doubao structuring | `doubao-seed-2-0-lite-260215` |
| Disagreement arbitration | `deepseek-v4-pro` |
| Embeddings | `text-embedding-v4` |

B1 clusters complete records. T1–T8 cluster selected semantic fields of hazard units. P1 performs object clustering within finding topics; P2 performs finding clustering within risk-setting topics. K1 builds topic relations and evaluates supporting units with model-generated semantic judgments.

See [usage](docs/usage.md) for multistage analysis, K1 and evaluation entry points. Algorithm and model parameters are in `config/`; prompt templates are in `prompts/`.

