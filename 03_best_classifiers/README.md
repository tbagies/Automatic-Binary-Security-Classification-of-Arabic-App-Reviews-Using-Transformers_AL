# 03 Best Classifiers

LoRA sentence representations and the five selected classical voters used by Active Learning.

Fine-tuning uses BatchHardTripletLoss, three epochs, batch size 16, learning rate 2e-5, and a 10% warmup ratio. LoRA uses rank 16, alpha 32, dropout 0.1, no bias adaptation, and targets the query, key, value, and dense modules.
