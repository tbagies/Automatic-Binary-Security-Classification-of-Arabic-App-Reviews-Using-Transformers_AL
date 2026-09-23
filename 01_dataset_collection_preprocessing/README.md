# 01 Dataset Collection and Preprocessing

Raw extracted sources remain immutable. Preprocessing removes English-only reviews, strips non-Arabic tokens and symbols from mixed reviews, retains only CAMeL/AraBERT-consistent negative or neutral sentiment, and then removes empty and duplicate text. Application/store and source filename metadata are retained in the 3,467,283-row review pool.

The dataset stages are 14,667,283 collected reviews, 7,037,690 Arabic reviews, 3,762,702 consistently negative or neutral reviews, and 3,467,283 reviews after removing empty and duplicate text. These stages are recorded in `provenance/manuscript_dataset_claims.json`.
