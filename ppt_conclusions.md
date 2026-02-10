# Key Conclusions for CAN Bus Intrusion Detection

## Model Performance Summary

- **Random Forest (RF) Significantly Outperforms Logistic Regression (LR)**
  - Average Accuracy: RF 94.2% vs LR 87.8% (+6.4%)
  - Average Recall: RF 82.3% vs LR 6.3% (+76.0%)
  - Average F1-Score: RF 82.7% vs LR 10.8% (+72.0%)

## Attack Detection Performance

### ✅ Fuzzing Attacks (Easiest)
- High detectability due to anomalous random/invalid data patterns

### ⚠️ Replay Attacks (Moderate)
- Challenge: Valid messages injected at wrong times

### 🔴 Combined Attacks (Hardest)
- Most realistic attack scenario — mix of replay and fuzzing

## Key Takeaways

1. **Use Random Forest** with `class_weight='balanced'`
2. **Avoid Logistic Regression** — fails on replay and combined attacks
3. **Recall is critical** — missed attacks are more dangerous than false alarms
4. **Attack complexity matters** — combined attacks need special attention

---

**Conclusion**: Random Forest achieves 94% average accuracy and 82% average recall, significantly outperforming Logistic Regression (88% accuracy, 6% recall).
