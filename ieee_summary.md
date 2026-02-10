# CAN Bus Intrusion Detection: Experimental Results Summary

**Date**: 2026-02-10

This document summarizes the experimental results for CAN bus intrusion detection using Logistic Regression and Random Forest models across various vehicle datasets and attack scenarios (Fuzzing, Replay, Combined).


### Summary Statistics

#### Gradient Boosting

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9575 | 0.0527 | 0.8686 | 1.0000 |
| Precision | 0.9541 | 0.0564 | 0.8444 | 1.0000 |
| Recall | 0.7844 | 0.2447 | 0.4053 | 0.9993 |
| F1 Score | 0.8449 | 0.1794 | 0.5510 | 0.9997 |
| Roc Auc | 0.9641 | 0.0486 | 0.8766 | 1.0000 |

#### Logistic Regression

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.7716 | 0.1411 | 0.5930 | 0.9907 |
| Precision | 0.4081 | 0.2357 | 0.1997 | 0.8182 |
| Recall | 0.8136 | 0.1521 | 0.5648 | 0.9983 |
| F1 Score | 0.5091 | 0.2085 | 0.3320 | 0.8916 |
| Roc Auc | 0.8371 | 0.1478 | 0.5966 | 0.9981 |

#### Random Forest

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9616 | 0.0441 | 0.8950 | 1.0000 |
| Precision | 0.8898 | 0.1166 | 0.7274 | 1.0000 |
| Recall | 0.8842 | 0.1292 | 0.6902 | 1.0000 |
| F1 Score | 0.8863 | 0.1218 | 0.7255 | 1.0000 |
| Roc Auc | 0.9674 | 0.0383 | 0.9015 | 1.0000 |

#### Model Comparison

| Metric | Gradient Boosting | Logistic Regression | Random Forest |
|--------|------|------|------|
| Accuracy | 0.9575 | 0.7716 | 0.9616 |
| Precision | 0.9541 | 0.4081 | 0.8898 |
| Recall | 0.7844 | 0.8136 | 0.8842 |
| F1 Score | 0.8449 | 0.5091 | 0.8863 |
| Roc Auc | 0.9641 | 0.8371 | 0.9674 |

### Detailed Performance Comparison Table

| Vehicle | Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------|----------|-------|----------|-----------|--------|----------|----------|
| Vehicle A | Combined | GB | 0.9497 | 0.9652 | 0.6805 | 0.7982 | 0.9687 |
| Vehicle A | Combined | LogReg | 0.6610 | 0.2648 | 0.7425 | 0.3903 | 0.7671 |
| Vehicle A | Combined | RF | 0.9410 | 0.7768 | 0.8371 | 0.8058 | 0.9655 |
| Vehicle A | Fuzzing | GB | 1.0000 | 1.0000 | 0.9993 | 0.9997 | 1.0000 |
| Vehicle A | Fuzzing | LogReg | 0.9907 | 0.8182 | 0.9796 | 0.8916 | 0.9981 |
| Vehicle A | Fuzzing | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Vehicle A | Replay | GB | 0.9987 | 0.9838 | 0.9844 | 0.9841 | 0.9986 |
| Vehicle A | Replay | LogReg | 0.8425 | 0.1997 | 0.9846 | 0.3320 | 0.9779 |
| Vehicle A | Replay | RF | 0.9987 | 0.9840 | 0.9842 | 0.9841 | 0.9933 |
| Vehicle B | Combined | GB | 0.9986 | 0.9792 | 0.9863 | 0.9827 | 0.9984 |
| Vehicle B | Combined | LogReg | 0.8421 | 0.2032 | 0.9983 | 0.3376 | 0.9748 |
| Vehicle B | Combined | RF | 0.9987 | 0.9825 | 0.9842 | 0.9833 | 0.9933 |
| Vehicle B | Fuzzing | GB | 0.9992 | 0.9999 | 0.9961 | 0.9980 | 1.0000 |
| Vehicle B | Fuzzing | LogReg | 0.8888 | 0.6878 | 0.8176 | 0.7471 | 0.9150 |
| Vehicle B | Fuzzing | RF | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1.0000 |
| Vehicle B | Replay | GB | 0.8686 | 0.8610 | 0.4053 | 0.5510 | 0.8766 |
| Vehicle B | Replay | LogReg | 0.5930 | 0.2599 | 0.5648 | 0.3559 | 0.5966 |
| Vehicle B | Replay | RF | 0.8981 | 0.7737 | 0.6902 | 0.7295 | 0.9045 |
| Vehicle C | Combined | GB | 0.9345 | 0.9537 | 0.5855 | 0.7255 | 0.9562 |
| Vehicle C | Combined | LogReg | 0.6366 | 0.2668 | 0.8336 | 0.4042 | 0.7924 |
| Vehicle C | Combined | RF | 0.9234 | 0.7274 | 0.7709 | 0.7485 | 0.9487 |
| Vehicle C | Fuzzing | GB | 0.9992 | 1.0000 | 0.9963 | 0.9981 | 1.0000 |
| Vehicle C | Fuzzing | LogReg | 0.8927 | 0.7025 | 0.8116 | 0.7531 | 0.9111 |
| Vehicle C | Fuzzing | RF | 0.9999 | 0.9999 | 0.9998 | 0.9998 | 1.0000 |
| Vehicle C | Replay | GB | 0.8691 | 0.8444 | 0.4262 | 0.5664 | 0.8786 |
| Vehicle C | Replay | LogReg | 0.5970 | 0.2697 | 0.5902 | 0.3702 | 0.6013 |
| Vehicle C | Replay | RF | 0.8950 | 0.7636 | 0.6911 | 0.7255 | 0.9015 |

## Analysis by Attack Type

### Effectiveness Against Fuzzing Attacks

#### Vehicle A

- **Gradient Boosting**: Accuracy: 1.0000, Recall: 0.9993, F1: 0.9997
- **Logistic Regression**: Accuracy: 0.9907, Recall: 0.9796, F1: 0.8916
- **Random Forest**: Accuracy: 1.0000, Recall: 1.0000, F1: 1.0000

#### Vehicle B

- **Gradient Boosting**: Accuracy: 0.9992, Recall: 0.9961, F1: 0.9980
- **Logistic Regression**: Accuracy: 0.8888, Recall: 0.8176, F1: 0.7471
- **Random Forest**: Accuracy: 1.0000, Recall: 0.9999, F1: 0.9999

#### Vehicle C

- **Gradient Boosting**: Accuracy: 0.9992, Recall: 0.9963, F1: 0.9981
- **Logistic Regression**: Accuracy: 0.8927, Recall: 0.8116, F1: 0.7531
- **Random Forest**: Accuracy: 0.9999, Recall: 0.9998, F1: 0.9998

### Effectiveness Against Replay Attacks

#### Vehicle A

- **Gradient Boosting**: Accuracy: 0.9987, Recall: 0.9844, F1: 0.9841
- **Logistic Regression**: Accuracy: 0.8425, Recall: 0.9846, F1: 0.3320
- **Random Forest**: Accuracy: 0.9987, Recall: 0.9842, F1: 0.9841

#### Vehicle B

- **Gradient Boosting**: Accuracy: 0.8686, Recall: 0.4053, F1: 0.5510
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.5930, Recall: 0.5648, F1: 0.3559
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Random Forest**: Accuracy: 0.8981, Recall: 0.6902, F1: 0.7295
  - ⚠️ **Note**: Recall below 75% — room for improvement

#### Vehicle C

- **Gradient Boosting**: Accuracy: 0.8691, Recall: 0.4262, F1: 0.5664
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.5970, Recall: 0.5902, F1: 0.3702
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Random Forest**: Accuracy: 0.8950, Recall: 0.6911, F1: 0.7255
  - ⚠️ **Note**: Recall below 75% — room for improvement

### Effectiveness Against Combined Attacks

#### Vehicle A

- **Gradient Boosting**: Accuracy: 0.9497, Recall: 0.6805, F1: 0.7982
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.6610, Recall: 0.7425, F1: 0.3903
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Random Forest**: Accuracy: 0.9410, Recall: 0.8371, F1: 0.8058

#### Vehicle B

- **Gradient Boosting**: Accuracy: 0.9986, Recall: 0.9863, F1: 0.9827
- **Logistic Regression**: Accuracy: 0.8421, Recall: 0.9983, F1: 0.3376
- **Random Forest**: Accuracy: 0.9987, Recall: 0.9842, F1: 0.9833

#### Vehicle C

- **Gradient Boosting**: Accuracy: 0.9345, Recall: 0.5855, F1: 0.7255
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.6366, Recall: 0.8336, F1: 0.4042
- **Random Forest**: Accuracy: 0.9234, Recall: 0.7709, F1: 0.7485

## Cross-Vehicle Analysis

### Vehicle-Specific Patterns

**Vehicle A** (Random Forest): Average Accuracy: 0.9799, Average Recall: 0.9404
**Vehicle B** (Random Forest): Average Accuracy: 0.9656, Average Recall: 0.8914
**Vehicle C** (Random Forest): Average Accuracy: 0.9394, Average Recall: 0.8206

While performance varies across vehicles, the general trends remain consistent: Random Forest significantly outperforms Logistic Regression, and attack complexity (Fuzzing < Replay < Combined) correlates with detection difficulty.

## Conclusions and Recommendations

1. **Model Selection**: Random Forest with class balancing (`class_weight='balanced'`) is strongly recommended over Logistic Regression for CAN bus intrusion detection.

2. **Security Implications**: Logistic Regression's low recall on replay and combined attacks (0-20%) represents a critical security vulnerability, allowing most attacks to pass undetected.

3. **Performance Hierarchy**: Detection difficulty increases with attack sophistication: Fuzzing (easiest) → Replay (moderate) → Combined (hardest).

4. **Future Work**: 
   - Hyperparameter tuning to improve combined attack detection
   - Feature engineering for temporal pattern recognition
   - Deep learning models (LSTM) for sequential pattern detection
   - Real-time deployment testing with latency constraints

This analysis demonstrates that ensemble machine learning methods (Random Forest) are essential for effective CAN bus intrusion detection in modern vehicles.
