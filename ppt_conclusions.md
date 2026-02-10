# Key Conclusions for CAN Bus Intrusion Detection

## Model Performance Summary

- **Gradient Boosting**: Accuracy 95.7%, Recall 78.2%, F1 84.3%
- **Logistic Regression**: Accuracy 90.0%, Recall 29.5%, F1 34.9%
- **Random Forest**: Accuracy 96.1%, Recall 88.2%, F1 88.5%

## Attack Detection Performance

### ✅ Fuzzing Attacks (Easiest)
- High detectability due to anomalous random/invalid data patterns

### ⚠️ Replay Attacks (Moderate)
- Challenge: Valid messages injected at wrong times

### 🔴 Combined Attacks (Hardest)
- Most realistic attack scenario — mix of replay and fuzzing

## Key Takeaways

1. **Use ensemble methods** (Random Forest / Gradient Boosting) with `class_weight='balanced'`
2. **Avoid Logistic Regression** — fails on replay and combined attacks
3. **Recall is critical** — missed attacks are more dangerous than false alarms
4. **Attack complexity matters** — combined attacks need special attention

---

**Conclusion**: Random Forest achieves 96% average accuracy and 88% average recall, significantly outperforming Logistic Regression (90% accuracy, 30% recall).
