# Captured Table 19 Outputs

These files reproduce manuscript Table 19 and Figure 5 exactly after rounding to two decimal percentage points.

- CAMeLBERT: accuracy 0.9575808878856283; matrix 5010/306/145/5171.
- MARBERTv2: accuracy 0.95372460496614; matrix 5014/302/190/5126.
- Paraphrase multilingual MiniLM-L12-v2: accuracy 0.9499623777276147; matrix 4998/318/214/5102.

The runs used the 10,632-row balanced training snapshot, `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, BatchAllTripletLoss, 10 epochs, batch size 128, learning rate 2e-5, 10% warmup, and cosine 1-nearest-neighbor classification within each fold.
