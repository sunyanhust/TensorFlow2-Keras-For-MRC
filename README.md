# TensorFlow2 Keras For MRC Baseline
Machine Reading Comprehension

## SQuAD

BERT Paprer  Dev  Dataset EM:**80.8**  F1:**88.5** 

|       Model       | Device | Batch Size | Leanring Rate | Optimizer |  EM  |  F1  |
| :---------------: | :----: | :--------: | :-----------: | :-------: | :--: | :--: |
| bert-base-uncased |  GPU   |     6      |     5e-5      |   Adam    | 73.8 | 82.4 |
| bert-base-uncased |  TPU   |     6      |     5e-5      |   Adam    | 72.5 | 81.7 |
| bert-base-uncased |  TPU   |     6      |     3e-5      |   Adam    | 78.5 | 85.4 |
| bert-base-uncased |  TPU   |     64     |     5e-5      |   Adam    | 76.4 | 84.6 |
| bert-base-uncased |  TPU   |    128     |     5e-5      |   Adam    | 78.2 | 86.0 |
| bert-base-uncased |  TPU   |    128     |     3e-5      |   Adam    | 78.5 | 86.5 |
| bert-base-uncased |  TPU   |    128     |     3e-5      |   Nadam   | 79.3 | 86.9 |

## TODO List
- [ ] Concat BERT last three  layers sequence output.
- [ ] Mean five time output result. 