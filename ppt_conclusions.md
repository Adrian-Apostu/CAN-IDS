# Key Conclusions for CAN Bus Intrusion Detection

## Model Performance Summary

- **Random Forest (RF) Significantly Outperforms Logistic Regression (LR)**
  - Average Accuracy: RF 96.1% vs LR 90.0% (+6.1%)
  - Average Recall: RF 88.2% vs LR 29.5% (+58.7%)
  - Average F1-Score: RF 88.5% vs LR 34.9% (+53.5%)

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

**Conclusion**: Random Forest achieves 96% average accuracy and 88% average recall, significantly outperforming Logistic Regression (90% accuracy, 29% recall).
