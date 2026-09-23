# 05 Fine-tune Cross Validation

Fresh SentenceTransformer encoders are trained inside each of ten stratified folds with BatchAllTripletLoss (10 epochs, batch size 128, learning rate 2e-5, 10% warmup, and a linear learning-rate scheduler). Each held-out fold is classified by cosine 1-nearest-neighbor against that fold's training reviews. The script generates one out-of-fold prediction for every review.
