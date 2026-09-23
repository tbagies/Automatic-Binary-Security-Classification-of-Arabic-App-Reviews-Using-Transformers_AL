# 07 Cross Validation Results

Aggregate fold metrics, summary metrics, and confusion matrices for four TF-IDF, five LoRA-feature, and three fine-tuned pipelines.

The Table 19 metrics and Figure 5 matrices are under `captured_table_19`, and the generic `confusion_matrices/fine_tuned_*` files represent those publication outputs. `metrics/manuscript_table_19_finetuned_metrics.csv` and `metrics/manuscript_reported_confusion_matrices.csv` provide compact manuscript tables.

`metrics/manuscript_table_22_cross_validation_metrics.csv` contains the comprehensive 12-pipeline Table 22. `metrics/cross_validation_summary_metrics.csv` and the confusion matrices provide the corresponding aggregate results.

The Table 19 CSVs use positive-class (`security`) precision, recall, and F1. Table 22 reports macro-averaged metrics.
