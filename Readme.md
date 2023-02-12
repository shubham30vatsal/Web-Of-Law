# Classification of US Supreme Court Cases using BERT-based Techniques



## This repositiory aims to experiment with new BERT-based techniques for the classification of long (over 512 tokens) US Supreme Court documents. The details have been listed below:

1. BEST-512 folder contains all the experiments done with different BERT-based models and aims at identifying the best 512 token chunk with respect to the two classification tasks. Please read the paper for more details.
2. Concat folder showcases how pooling of CLS tokens from parallel BERT-based models works out with respect to the two classification tasks. Please read the paper for more details.
3. Ensemble folder experiments how maximum voting strategy from parallel BERT-based models works out with respect to the two classification tasks. Please read the paper for more details.
4. LSM folder experiments with models like Longformer which can accept input sequences longer than 512 tokens and compares results with other techniques for the two classification tasks. Please read the paper for more details.
5. Stride folder shows how a stride window (contextual overlap between two parallel BERT-based models) can affect the results for the two classification tasks. Please read the paper for more details.
6. Summarization folder contains all the experiments done with different BERT-based models and aims at summarizing the documents over 512 tokens to <= 512 tokens for the two classification tasks. Please read the paper for more details.
7. label_sc.txt contains the labels for all the data points for the 15-label classification task.
8. label_sc_279.txt contains the labels for all the data points for the 279-label classification task.



Please contact sv2128@nyu.edu for any issues/bugs.
