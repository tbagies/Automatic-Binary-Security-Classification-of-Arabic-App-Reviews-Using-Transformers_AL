# Arabic Mobile App Reviews: Security Classification

This repository contains the datasets, code, models, and results for binary classification of Arabic mobile-app reviews as security or non-security. Application name and store are preserved for every review-level record.

## Workflow

1. `01_dataset_collection_preprocessing` - extracted source data and prepared review pools.
2. `02_initial_expert_annotation` - the 320 expert-reviewed workbook and balanced 300-row seed.
3. `03_best_classifiers` - LoRA sentence representations and the five selected classical voters.
4. `04_active_learning` - ten unanimous-agreement and human-validation iterations.
5. `05_finetune_cross_validation` - fold-specific SentenceTransformer training and 1-nearest-neighbor evaluation.
6. `06_full_dataset_finetuning` - CAMeLBERT, MARBERTv2, and Paraphrase embedding models trained on all 10,632 rows.
7. `07_cross_validation_results` - 12-pipeline cross-validation results.
8. `08_heldout_evaluation` - disjoint 1,000-row dataset and aggregate evaluation results.
9. `09_significance_tests` - McNemar and paired tests.
10. `10_error_analysis` - cross-validation and heldout mistakes with application/store metadata.
11. `11_wgi_analysis` - qualitative web-interface examples and captures.
12. `12_llm_gradio` - CAMeLBERT classification, Qwen3-8B explanations, and Gradio application.

Python under each `code/python` directory is canonical. The matching `code/html` tree is generated without executing training. Existing verified outputs are referenced by their hashes in HTML metadata.

## Reproducibility

- When regenerated, cross-validation predictions are produced only by models trained without that fold's test rows.
- Models in `06_full_dataset_finetuning/models` are for heldout inference and the WGI/LLM stage, never out-of-fold prediction.
- TF-IDF pipelines are fitted baselines; they are not transformer fine-tuning runs.
- Active Learning selects candidates only through five-voter unanimity and reproducible random sampling within each app/store and class.
- Active Learning uses three epochs, learning rate 2e-5, 10% warmup, BatchHardTripletLoss, and LoRA rank 16, alpha 32, and dropout 0.1.
- Aggregate metrics and p-value matrices have no row source. Every review-level CSV retains `app_name` and `store`.
- Dataset stages and counts are recorded in `provenance/manuscript_dataset_claims.json`.

Run `python tools/verify_repository.py` for the complete lightweight integrity suite. Heavy GPU training is intentionally opt-in.
