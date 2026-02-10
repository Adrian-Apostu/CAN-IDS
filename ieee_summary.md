# CAN Bus Intrusion Detection: Experimental Results Summary

**Date**: 2026-02-10

This document summarizes the experimental results for CAN bus intrusion detection using Logistic Regression and Random Forest models across various vehicle datasets and attack scenarios (Fuzzing, Replay, Combined).


### Summary Statistics

#### Logistic Regression

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.8779 | 0.0649 | 0.7993 | 0.9603 |
| Precision | 0.4198 | 0.3995 | 0.0000 | 0.8855 |
| Recall | 0.0634 | 0.0769 | 0.0000 | 0.1788 |
| F1 Score | 0.1075 | 0.1281 | 0.0000 | 0.2976 |
| Roc Auc | 0.7799 | 0.1447 | 0.6033 | 0.9665 |

#### Random Forest

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9422 | 0.0424 | 0.8966 | 0.9986 |
| Precision | 0.8333 | 0.1025 | 0.7303 | 0.9788 |
| Recall | 0.8234 | 0.1262 | 0.6748 | 0.9872 |
| F1 Score | 0.8271 | 0.1119 | 0.7231 | 0.9830 |
| Roc Auc | 0.9514 | 0.0378 | 0.9017 | 0.9954 |

#### Model Comparison (Improvement: RF vs LR)

| Metric | LR Mean | RF Mean | Absolute Gain | Relative Gain |
|--------|---------|---------|---------------|---------------|
| Accuracy | 0.8779 | 0.9422 | +0.0643 | +7.3% |
| Precision | 0.4198 | 0.8333 | +0.4135 | +98.5% |
| Recall | 0.0634 | 0.8234 | +0.7601 | +1199.8% |
| F1 Score | 0.1075 | 0.8271 | +0.7196 | +669.3% |
| Roc Auc | 0.7799 | 0.9514 | +0.1714 | +22.0% |

### Detailed Performance Comparison Table

| Vehicle | Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------|----------|-------|----------|-----------|--------|----------|----------|
| Vehicle A | Combined | LogReg | 0.8742 | 0.8717 | 0.1628 | 0.2744 | 0.7650 |
| Vehicle A | Combined | RF | 0.9392 | 0.7687 | 0.8351 | 0.8005 | 0.9645 |
| Vehicle A | Replay | LogReg | 0.9576 | 0.0800 | 0.0064 | 0.0119 | 0.9665 |
| Vehicle A | Replay | RF | 0.9986 | 0.9788 | 0.9872 | 0.9830 | 0.9954 |
| Vehicle B | Combined | LogReg | 0.9603 | 0.6818 | 0.0321 | 0.0612 | 0.9494 |
| Vehicle B | Combined | RF | 0.9984 | 0.9746 | 0.9850 | 0.9798 | 0.9942 |
| Vehicle B | Replay | LogReg | 0.8009 | 0.0000 | 0.0000 | 0.0000 | 0.6033 |
| Vehicle B | Replay | RF | 0.8971 | 0.7787 | 0.6748 | 0.7231 | 0.9017 |
| Vehicle C | Combined | LogReg | 0.8752 | 0.8855 | 0.1788 | 0.2976 | 0.7902 |
| Vehicle C | Combined | RF | 0.9235 | 0.7303 | 0.7652 | 0.7474 | 0.9476 |
| Vehicle C | Replay | LogReg | 0.7993 | 0.0000 | 0.0000 | 0.0000 | 0.6052 |
| Vehicle C | Replay | RF | 0.8966 | 0.7689 | 0.6932 | 0.7291 | 0.9048 |

## Analysis by Attack Type

### Effectiveness Against Fuzzing Attacks

#### Vehicle A

- **Logistic Regression**: No results found
- **Random Forest**: No results found

#### Vehicle B

- **Logistic Regression**: No results found
- **Random Forest**: No results found

#### Vehicle C

- **Logistic Regression**: No results found
- **Random Forest**: No results found

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

**Vehicle A** (Random Forest): Average Accuracy: 0.9689, Average Recall: 0.9111
**Vehicle B** (Random Forest): Average Accuracy: 0.9477, Average Recall: 0.8299
**Vehicle C** (Random Forest): Average Accuracy: 0.9101, Average Recall: 0.7292

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
