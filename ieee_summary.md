# CAN Bus Intrusion Detection: Experimental Results Summary

**Date**: 2026-02-10

This document summarizes the experimental results for CAN bus intrusion detection using Logistic Regression and Random Forest models across various vehicle datasets and attack scenarios (Fuzzing, Replay, Combined).


### Summary Statistics

#### Gradient Boosting

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9573 | 0.0525 | 0.8673 | 1.0000 |
| Precision | 0.9548 | 0.0537 | 0.8506 | 1.0000 |
| Recall | 0.7818 | 0.2476 | 0.4045 | 0.9989 |
| F1 Score | 0.8426 | 0.1800 | 0.5483 | 0.9995 |
| Roc Auc | 0.9643 | 0.0474 | 0.8782 | 1.0000 |

#### Logistic Regression

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9002 | 0.0653 | 0.7993 | 0.9973 |
| Precision | 0.5899 | 0.4073 | 0.0000 | 0.9864 |
| Recall | 0.2951 | 0.3421 | 0.0000 | 0.9435 |
| F1 Score | 0.3493 | 0.3616 | 0.0000 | 0.9644 |
| Roc Auc | 0.8321 | 0.1412 | 0.6034 | 0.9969 |

#### Random Forest

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9614 | 0.0440 | 0.8965 | 1.0000 |
| Precision | 0.8887 | 0.1151 | 0.7294 | 1.0000 |
| Recall | 0.8823 | 0.1323 | 0.6748 | 1.0000 |
| F1 Score | 0.8846 | 0.1224 | 0.7238 | 1.0000 |
| Roc Auc | 0.9676 | 0.0385 | 0.9017 | 1.0000 |

#### Model Comparison

| Metric | Gradient Boosting | Logistic Regression | Random Forest |
|--------|------|------|------|
| Accuracy | 0.9573 | 0.9002 | 0.9614 |
| Precision | 0.9548 | 0.5899 | 0.8887 |
| Recall | 0.7818 | 0.2951 | 0.8823 |
| F1 Score | 0.8426 | 0.3493 | 0.8846 |
| Roc Auc | 0.9643 | 0.8321 | 0.9676 |

### Detailed Performance Comparison Table

| Vehicle | Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------|----------|-------|----------|-----------|--------|----------|----------|
| Vehicle A | Combined | GB | 0.9507 | 0.9693 | 0.6841 | 0.8021 | 0.9684 |
| Vehicle A | Combined | LogReg | 0.8742 | 0.8729 | 0.1628 | 0.2745 | 0.7650 |
| Vehicle A | Combined | RF | 0.9388 | 0.7667 | 0.8356 | 0.7997 | 0.9646 |
| Vehicle A | Fuzzing | GB | 1.0000 | 1.0000 | 0.9989 | 0.9995 | 1.0000 |
| Vehicle A | Fuzzing | LogReg | 0.9973 | 0.9864 | 0.9435 | 0.9644 | 0.9969 |
| Vehicle A | Fuzzing | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Vehicle A | Replay | GB | 0.9986 | 0.9778 | 0.9872 | 0.9825 | 0.9988 |
| Vehicle A | Replay | LogReg | 0.9573 | 0.0741 | 0.0064 | 0.0118 | 0.9665 |
| Vehicle A | Replay | RF | 0.9986 | 0.9788 | 0.9872 | 0.9830 | 0.9954 |
| Vehicle B | Combined | GB | 0.9984 | 0.9726 | 0.9872 | 0.9799 | 0.9982 |
| Vehicle B | Combined | LogReg | 0.9603 | 0.6818 | 0.0321 | 0.0612 | 0.9493 |
| Vehicle B | Combined | RF | 0.9984 | 0.9746 | 0.9850 | 0.9798 | 0.9942 |
| Vehicle B | Fuzzing | GB | 0.9991 | 0.9989 | 0.9968 | 0.9979 | 1.0000 |
| Vehicle B | Fuzzing | LogReg | 0.9183 | 0.9041 | 0.6636 | 0.7654 | 0.9077 |
| Vehicle B | Fuzzing | RF | 0.9999 | 1.0000 | 0.9996 | 0.9998 | 1.0000 |
| Vehicle B | Replay | GB | 0.8673 | 0.8506 | 0.4045 | 0.5483 | 0.8782 |
| Vehicle B | Replay | LogReg | 0.8009 | 0.0000 | 0.0000 | 0.0000 | 0.6034 |
| Vehicle B | Replay | RF | 0.8975 | 0.7803 | 0.6748 | 0.7238 | 0.9017 |
| Vehicle C | Combined | GB | 0.9304 | 0.9587 | 0.5531 | 0.7015 | 0.9517 |
| Vehicle C | Combined | LogReg | 0.8752 | 0.8855 | 0.1788 | 0.2976 | 0.7902 |
| Vehicle C | Combined | RF | 0.9234 | 0.7294 | 0.7655 | 0.7470 | 0.9476 |
| Vehicle C | Fuzzing | GB | 0.9990 | 1.0000 | 0.9953 | 0.9976 | 1.0000 |
| Vehicle C | Fuzzing | LogReg | 0.9190 | 0.9045 | 0.6687 | 0.7689 | 0.9034 |
| Vehicle C | Fuzzing | RF | 0.9999 | 1.0000 | 0.9995 | 0.9998 | 1.0000 |
| Vehicle C | Replay | GB | 0.8720 | 0.8655 | 0.4292 | 0.5738 | 0.8837 |
| Vehicle C | Replay | LogReg | 0.7993 | 0.0000 | 0.0000 | 0.0000 | 0.6062 |
| Vehicle C | Replay | RF | 0.8965 | 0.7683 | 0.6932 | 0.7288 | 0.9049 |

## Analysis by Attack Type

### Effectiveness Against Fuzzing Attacks

#### Vehicle A

- **Gradient Boosting**: Accuracy: 1.0000, Recall: 0.9989, F1: 0.9995
- **Logistic Regression**: Accuracy: 0.9973, Recall: 0.9435, F1: 0.9644
- **Random Forest**: Accuracy: 1.0000, Recall: 1.0000, F1: 1.0000

#### Vehicle B

- **Gradient Boosting**: Accuracy: 0.9991, Recall: 0.9968, F1: 0.9979
- **Logistic Regression**: Accuracy: 0.9183, Recall: 0.6636, F1: 0.7654
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Random Forest**: Accuracy: 0.9999, Recall: 0.9996, F1: 0.9998

#### Vehicle C

- **Gradient Boosting**: Accuracy: 0.9990, Recall: 0.9953, F1: 0.9976
- **Logistic Regression**: Accuracy: 0.9190, Recall: 0.6687, F1: 0.7689
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Random Forest**: Accuracy: 0.9999, Recall: 0.9995, F1: 0.9998

### Effectiveness Against Replay Attacks

#### Vehicle A

- **Gradient Boosting**: Accuracy: 0.9986, Recall: 0.9872, F1: 0.9825
- **Logistic Regression**: Accuracy: 0.9573, Recall: 0.0064, F1: 0.0118
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9986, Recall: 0.9872, F1: 0.9830

#### Vehicle B

- **Gradient Boosting**: Accuracy: 0.8673, Recall: 0.4045, F1: 0.5483
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.8009, Recall: 0.0000, F1: 0.0000
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.8975, Recall: 0.6748, F1: 0.7238
  - ⚠️ **Note**: Recall below 75% — room for improvement

#### Vehicle C

- **Gradient Boosting**: Accuracy: 0.8720, Recall: 0.4292, F1: 0.5738
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.7993, Recall: 0.0000, F1: 0.0000
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.8965, Recall: 0.6932, F1: 0.7288
  - ⚠️ **Note**: Recall below 75% — room for improvement

### Effectiveness Against Combined Attacks

#### Vehicle A

- **Gradient Boosting**: Accuracy: 0.9507, Recall: 0.6841, F1: 0.8021
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.8742, Recall: 0.1628, F1: 0.2745
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9388, Recall: 0.8356, F1: 0.7997

#### Vehicle B

- **Gradient Boosting**: Accuracy: 0.9984, Recall: 0.9872, F1: 0.9799
- **Logistic Regression**: Accuracy: 0.9603, Recall: 0.0321, F1: 0.0612
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9984, Recall: 0.9850, F1: 0.9798

#### Vehicle C

- **Gradient Boosting**: Accuracy: 0.9304, Recall: 0.5531, F1: 0.7015
  - ⚠️ **Note**: Recall below 75% — room for improvement
- **Logistic Regression**: Accuracy: 0.8752, Recall: 0.1788, F1: 0.2976
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9234, Recall: 0.7655, F1: 0.7470

## Cross-Vehicle Analysis

### Vehicle-Specific Patterns

**Vehicle A** (Random Forest): Average Accuracy: 0.9791, Average Recall: 0.9409
**Vehicle B** (Random Forest): Average Accuracy: 0.9653, Average Recall: 0.8865
**Vehicle C** (Random Forest): Average Accuracy: 0.9399, Average Recall: 0.8194

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
