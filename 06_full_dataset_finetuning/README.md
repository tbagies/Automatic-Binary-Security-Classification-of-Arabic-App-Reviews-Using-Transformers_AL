# 06 Full Dataset Fine-tuning

The three saved SentenceTransformer embedding models were trained on the complete balanced 10,632-row dataset with BatchAllTripletLoss (10 epochs, batch size 128, learning rate 2e-5, 10% warmup, and the trainer's linear learning-rate scheduler). Downstream heldout and WGI predictions use cosine 1-nearest-neighbor against the labeled training embeddings.
