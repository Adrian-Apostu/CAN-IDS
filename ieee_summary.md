# CAN Bus Intrusion Detection: Experimental Results Summary

**Date**: 2026-02-10

This document summarizes the experimental results for CAN bus intrusion detection using Logistic Regression and Random Forest models across various vehicle datasets and attack scenarios (Fuzzing, Replay, Combined).


### Summary Statistics

#### Logistic Regression

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.8913 | 0.0660 | 0.7993 | 0.9972 |
| Precision | 0.5222 | 0.4122 | 0.0000 | 0.9863 |
| Recall | 0.2023 | 0.2922 | 0.0000 | 0.9424 |
| F1 Score | 0.2525 | 0.3146 | 0.0000 | 0.9639 |
| Roc Auc | 0.8111 | 0.1450 | 0.6033 | 0.9969 |

#### Random Forest

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9538 | 0.0444 | 0.8966 | 1.0000 |
| Precision | 0.8667 | 0.1134 | 0.7303 | 1.0000 |
| Recall | 0.8587 | 0.1331 | 0.6748 | 1.0000 |
| F1 Score | 0.8617 | 0.1216 | 0.7231 | 1.0000 |
| Roc Auc | 0.9611 | 0.0390 | 0.9017 | 1.0000 |

#### Model Comparison (Improvement: RF vs LR)

| Metric | LR Mean | RF Mean | Absolute Gain | Relative Gain |
|--------|---------|---------|---------------|---------------|
| Accuracy | 0.8913 | 0.9538 | +0.0625 | +7.0% |
| Precision | 0.5222 | 0.8667 | +0.3445 | +66.0% |
| Recall | 0.2023 | 0.8587 | +0.6564 | +324.5% |
| F1 Score | 0.2525 | 0.8617 | +0.6092 | +241.2% |
| Roc Auc | 0.8111 | 0.9611 | +0.1500 | +18.5% |

### Detailed Performance Comparison Table

| Vehicle | Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------|----------|-------|----------|-----------|--------|----------|----------|
| Vehicle A | Combined | LogReg | 0.8742 | 0.8717 | 0.1628 | 0.2744 | 0.7650 |
| Vehicle A | Combined | LogReg | 0.8742 | 0.8717 | 0.1628 | 0.2744 | 0.7650 |
| Vehicle A | Combined | RF | 0.9392 | 0.7687 | 0.8351 | 0.8005 | 0.9645 |
| Vehicle A | Combined | RF | 0.9392 | 0.7687 | 0.8351 | 0.8005 | 0.9645 |
| Vehicle A | Fuzzing | LogReg | 0.9972 | 0.9863 | 0.9424 | 0.9639 | 0.9969 |
| Vehicle A | Fuzzing | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Vehicle A | Replay | LogReg | 0.9576 | 0.0800 | 0.0064 | 0.0119 | 0.9665 |
| Vehicle A | Replay | LogReg | 0.9576 | 0.0800 | 0.0064 | 0.0119 | 0.9665 |
| Vehicle A | Replay | RF | 0.9986 | 0.9788 | 0.9872 | 0.9830 | 0.9954 |
| Vehicle A | Replay | RF | 0.9986 | 0.9788 | 0.9872 | 0.9830 | 0.9954 |
| Vehicle B | Combined | LogReg | 0.9603 | 0.6818 | 0.0321 | 0.0612 | 0.9494 |
| Vehicle B | Combined | LogReg | 0.9603 | 0.6818 | 0.0321 | 0.0612 | 0.9494 |
| Vehicle B | Combined | RF | 0.9984 | 0.9746 | 0.9850 | 0.9798 | 0.9942 |
| Vehicle B | Combined | RF | 0.9984 | 0.9746 | 0.9850 | 0.9798 | 0.9942 |
| Vehicle B | Fuzzing | LogReg | 0.9182 | 0.9041 | 0.6632 | 0.7651 | 0.9077 |
| Vehicle B | Fuzzing | RF | 0.9999 | 1.0000 | 0.9996 | 0.9998 | 1.0000 |
| Vehicle B | Replay | LogReg | 0.8009 | 0.0000 | 0.0000 | 0.0000 | 0.6033 |
| Vehicle B | Replay | LogReg | 0.8009 | 0.0000 | 0.0000 | 0.0000 | 0.6033 |
| Vehicle B | Replay | RF | 0.8971 | 0.7787 | 0.6748 | 0.7231 | 0.9017 |
| Vehicle B | Replay | RF | 0.8971 | 0.7787 | 0.6748 | 0.7231 | 0.9017 |
| Vehicle C | Combined | LogReg | 0.8752 | 0.8855 | 0.1788 | 0.2976 | 0.7902 |
| Vehicle C | Combined | LogReg | 0.8752 | 0.8855 | 0.1788 | 0.2976 | 0.7902 |
| Vehicle C | Combined | RF | 0.9235 | 0.7303 | 0.7652 | 0.7474 | 0.9476 |
| Vehicle C | Combined | RF | 0.9235 | 0.7303 | 0.7652 | 0.7474 | 0.9476 |
| Vehicle C | Fuzzing | LogReg | 0.9190 | 0.9045 | 0.6687 | 0.7689 | 0.9033 |
| Vehicle C | Fuzzing | RF | 0.9999 | 1.0000 | 0.9995 | 0.9998 | 1.0000 |
| Vehicle C | Replay | LogReg | 0.7993 | 0.0000 | 0.0000 | 0.0000 | 0.6052 |
| Vehicle C | Replay | LogReg | 0.7993 | 0.0000 | 0.0000 | 0.0000 | 0.6052 |
| Vehicle C | Replay | RF | 0.8966 | 0.7689 | 0.6932 | 0.7291 | 0.9048 |
| Vehicle C | Replay | RF | 0.8966 | 0.7689 | 0.6932 | 0.7291 | 0.9048 |

## Analysis by Attack Type

### Effectiveness Against Fuzzing Attacks

#### Vehicle A

- **Logistic Regression**: Accuracy: 0.9972, Recall: 0.9424, F1: 0.9639
- **Random Forest**: Accuracy: 1.0000, Recall: 1.0000, F1: 1.0000

#### Vehicle B

- **Logistic Regression**: Accuracy: 0.9182, Recall: 0.6632, F1: 0.7651
- **Random Forest**: Accuracy: 0.9999, Recall: 0.9996, F1: 0.9998

#### Vehicle C

- **Logistic Regression**: Accuracy: 0.9190, Recall: 0.6687, F1: 0.7689
- **Random Forest**: Accuracy: 0.9999, Recall: 0.9995, F1: 0.9998

### Effectiveness Against Replay Attacks

#### Vehicle A

- **Logistic Regression**: Accuracy: 0.9576, Recall: 0.0064, F1: 0.0119
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9986, Recall: 0.9872, F1: 0.9830

#### Vehicle B

- **Logistic Regression**: Accuracy: 0.8009, Recall: 0.0000, F1: 0.0000
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.8971, Recall: 0.6748, F1: 0.7231
  - ⚠️ **Note**: Recall below 75% — room for improvement

#### Vehicle C

- **Logistic Regression**: Accuracy: 0.7993, Recall: 0.0000, F1: 0.0000
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.8966, Recall: 0.6932, F1: 0.7291
  - ⚠️ **Note**: Recall below 75% — room for improvement

### Effectiveness Against Combined Attacks

#### Vehicle A

- **Logistic Regression**: Accuracy: 0.8742, Recall: 0.1628, F1: 0.2744
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9392, Recall: 0.8351, F1: 0.8005

#### Vehicle B

- **Logistic Regression**: Accuracy: 0.9603, Recall: 0.0321, F1: 0.0612
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9984, Recall: 0.9850, F1: 0.9798

#### Vehicle C

- **Logistic Regression**: Accuracy: 0.8752, Recall: 0.1788, F1: 0.2976
  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!
- **Random Forest**: Accuracy: 0.9235, Recall: 0.7652, F1: 0.7474

## Cross-Vehicle Analysis

### Vehicle-Specific Patterns

**Vehicle A** (Random Forest): Average Accuracy: 0.9751, Average Recall: 0.9289
**Vehicle B** (Random Forest): Average Accuracy: 0.9582, Average Recall: 0.8638
**Vehicle C** (Random Forest): Average Accuracy: 0.9280, Average Recall: 0.7833

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
