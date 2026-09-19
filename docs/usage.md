# Usage

Run commands from the repository root. Specify the input file and an output directory outside the source directory.

## Structuring, clustering and ablation

`scripts/run_experiment_pipeline.py` supports `llm-extract`, `unit-dedupe`, `cluster`, `name`, `ablation-extract`, `ablation-cluster` and `ablation-name`. The `all` stage runs the main structuring and clustering pipeline; ablation stages are selected explicitly. Use `--help` for input, output and cache options.

The default configuration is `config/experiment.yaml`. `--offline-cache-only` prevents external requests and requires a compatible local cache. Dataset columns are selected through `--id-column` and `--text-column`; Excel inputs also accept `--sheet-name`.

The fields are `finding` (problem manifestation), `object` (risk objects), `scene` (raw context), `risk_scene` (risk setting) and `loc_detail` (location detail).

## Multistage analysis

After the main pipeline produces assignments and an embedding cache:

```bash
python scripts/run_multistage_analysis.py --run-dir /path/to/local-runs/experiment --embedding-cache /path/to/embedding-cache.jsonl --output-dir /path/to/multistage-output
```

The runner reuses the main pipeline's structured units, T1/T2/T3 assignments and channel inputs. The `multistage` section of `config/experiment.yaml` controls parent eligibility and local UMAP/HDBSCAN parameters. It writes parent summaries, unit assignments and naming-request inputs locally.

## K1

`industrial_hazard_analysis.topic_control_synthesis` provides the K1 Python API. Parameters are in `config/k1.yaml`, which loads its prompt templates from `prompts/k1.yaml`.

| Step | Functions |
| --- | --- |
| Candidate pairs from topic centroids | `generate_candidate_pairs` |
| Topic-pair requests and responses | `build_relation_requests`, `flatten_relation_results` |
| Pairwise-positive topic groups | `build_provisional_groups` |
| Group synthesis | `build_synthesis_group_requests`, `final_controls_from_synthesis` |
| Unit evidence | `build_synthesis_unit_requests`, `flatten_synthesis_unit_results` |
| Control semantics and deterministic counts | `build_synthesis_knowledge_card_requests`, `assemble_synthesis_knowledge_cards` |

Candidate generation requires a DataFrame with `topic_id` and `parent_context`, plus a centroid matrix in the same row order. A topic centroid is the mean of its member embeddings followed by L2 normalization. Topic IDs must be unique; names do not participate in candidate selection.

Request builders use topic objects containing temporary topic/sample IDs, topic context and representative semantic fields. Unit evidence uses `unit_id`, `topic_id`, `finding`, `objects` (a Python list) and `risk_scene`. Inspect the request builders and JSON schemas for each task's exact contract. Supply de-identified inputs without enterprise names or personal contact information.

Build requests with `TopicControlPromptBuilder(load_topic_control_config("config/k1.yaml"))`. Store the resulting request dictionaries as JSONL outside the repository. To execute a prepared request batch:

```bash
python scripts/run_k1_requests.py --requests /path/to/requests.jsonl --output-dir /path/to/k1-output
```

The command uses `QWEN_API_KEY`, validates request identities and response schemas, and writes responses and cache files to the chosen local output directory. Its output records can be supplied to the next API step. `--offline-cache-only` uses the local cache without making external requests.

## Evaluation

| Purpose | Script |
| --- | --- |
| Extraction metrics against supplied annotations | `evaluate_reliability.py` |
| Compare extraction systems | `compare_extraction_systems.py` |
| Main clustering statistics | `summarize_clustering_results.py` |
| Topic separation | `analyze_topic_separation.py` |
| Unit-level adjusted mutual information | `analyze_unit_level_correspondence.py` |
| Ablation record alignment | `build_ablation_common_records.py` |
| Ablation statistics | `summarize_ablation_results.py` |

All scripts are under `scripts/` and accept input and output paths through command-line arguments.

## Prompts

All method prompts are in `prompts/`. Appendices A–D cover structuring, arbitration and scene separation. Appendix E provides the standard topic-naming template; `cluster_naming.yaml` contains the modular templates for global and second-level topics. `k1.yaml` contains the shared definitions, task instructions and output constraints for K1.

The naming and K1 configuration files reference their templates through `prompts_path`, resolved relative to the configuration file.
