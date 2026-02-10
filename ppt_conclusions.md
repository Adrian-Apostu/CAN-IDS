# Key Conclusions for CAN Bus Intrusion Detection

## Model Performance Summary

- **Random Forest (RF) Significantly Outperforms Logistic Regression (LR)**
  - Average Accuracy: RF 93.4% vs LR 87.6% (+5.8%)
  - Average Recall: RF 75.5% vs LR 7.8% (+67.6%)
  - Average F1-Score: RF 79.0% vs LR 13.1% (+65.8%)

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

**Conclusion**: Random Forest achieves 93% average accuracy and 75% average recall, significantly outperforming Logistic Regression (88% accuracy, 8% recall).
