# MRC
Machine Reading Comprehension

## SQuAD

BERT Paprer  Dev  Dataset EM:**80.8**  F1:**88.5** 

| Device | Batch Size | Leanring Rate | Optimizer |  EM  |  F1  |
| :----: | :--------: | :-----------: | :-------: | :--: | :--: |
|  GPU   | 6 | 5e-5 | Adam | 73.8 | 82.4 |
|  TPU   |     6      |     5e-5      |   Adam    | 72.5 | 81.7 |
|  TPU   |     6      |     3e-5      |   Adam    | 78.5 | 85.4 |
|  TPU   |     64     |     5e-5      |   Adam    | 76.4 | 84.6 |
|  TPU   |    128     |     5e-5      |   Adam    | 78.2 | 86.0 |
|  TPU   |    128     |     3e-5      |   Adam    | 78.5 | 86.5 |
|  TPU   |    128     |     3e-5      |   Nadam   | 79.3 | 86.9 |
|        |            |               |           |      |      |
|        |            |               |           |      |      |

