# Key Conclusions for CAN Bus Intrusion Detection

## Model Performance Summary

- **Random Forest (RF) Significantly Outperforms Logistic Regression (LR)**
  - Average Accuracy: RF 95.4% vs LR 89.1% (+6.2%)
  - Average Recall: RF 85.9% vs LR 20.2% (+65.6%)
  - Average F1-Score: RF 86.2% vs LR 25.3% (+60.9%)

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

**Conclusion**: Random Forest achieves 95% average accuracy and 86% average recall, significantly outperforming Logistic Regression (89% accuracy, 20% recall).
