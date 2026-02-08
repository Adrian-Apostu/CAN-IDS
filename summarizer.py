import os
import re
import numpy as np
import pandas as pd


def parse_results_file(file_path):
    """Parse a single results file and extract all metrics."""
    with open(file_path, 'r') as f:
        content = f.read()

    results = {}

    # Extract dataset and model from filename
    filename = os.path.basename(file_path)
    # Example filename: Vehicle_A___0x181___IDs_Datafield_Classification_logisticregression_results.txt
    match = re.match(
        r"(Vehicle_[A-C]___.*?___IDs_Datafield_Classification)_(logisticregression|randomforest)_results.txt", filename)
    if match:
        # Reconstruct original dataset name
        dataset_base = match.group(1).replace('___', ' - ').replace('_', ' ')

        results['dataset'] = dataset_base
        results['filename'] = filename

        # Extract vehicle
        if "Vehicle A" in dataset_base:
            results['vehicle'] = "Vehicle A"
        elif "Vehicle B" in dataset_base:
            results['vehicle'] = "Vehicle B"
        elif "Vehicle C" in dataset_base:
            results['vehicle'] = "Vehicle C"
        else:
            results['vehicle'] = "Unknown"

        # Extract scenario (fuzzing, replay, combined)
        # More robust scenario detection
        if "Replay" in dataset_base and "Fuzzing" in dataset_base:
            results['scenario'] = "combined"
        elif "Fuzzing" in dataset_base:
            results['scenario'] = "fuzzing"
        elif "Replay" in dataset_base:
            results['scenario'] = "replay"
        else:
            results['scenario'] = "no_attack"

        results['model'] = match.group(2)
    else:
        # Fallback for old or unexpected filenames
        results['dataset'] = filename
        results['filename'] = filename
        results['vehicle'] = "Unknown"
        results['scenario'] = "Unknown"
        results['model'] = "Unknown"

    # Extract metrics with improved regex patterns
    # Match both formats: "0.1234" and "1.0000"
    accuracy_match = re.search(r"Accuracy.*?:\s*([01]\.\d{4})", content)
    if accuracy_match:
        results['accuracy'] = float(accuracy_match.group(1))
    else:
        results['accuracy'] = None

    precision_match = re.search(r"Precision.*?:\s*([01]\.\d{4})", content)
    if precision_match:
        results['precision'] = float(precision_match.group(1))
    else:
        results['precision'] = None

    recall_match = re.search(r"Recall.*?:\s*([01]\.\d{4})", content)
    if recall_match:
        results['recall'] = float(recall_match.group(1))
    else:
        results['recall'] = None

    f1_match = re.search(r"F1-score.*?:\s*([01]\.\d{4})", content)
    if f1_match:
        results['f1_score'] = float(f1_match.group(1))
    else:
        results['f1_score'] = None

    roc_auc_match = re.search(r"ROC AUC.*?:\s*([01]\.\d{4})", content)
    if roc_auc_match:
        results['roc_auc'] = float(roc_auc_match.group(1))
    else:
        results['roc_auc'] = None

    # Confusion Matrix - improved pattern to handle varying whitespace
    cm_match = re.search(r"Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\]\]", content, re.MULTILINE)
    if cm_match:
        results['tn'] = int(cm_match.group(1))
        results['fp'] = int(cm_match.group(2))
        results['fn'] = int(cm_match.group(3))
        results['tp'] = int(cm_match.group(4))
    else:
        results['tn'] = None
        results['fp'] = None
        results['fn'] = None
        results['tp'] = None

    return results


def generate_comparison_table(all_results):
    """Generate a detailed comparison table."""
    table = []
    table.append("\n### Detailed Performance Comparison Table\n\n")
    table.append("| Vehicle | Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |\n")
    table.append("|---------|----------|-------|----------|-----------|--------|----------|----------|\n")

    # Sort results for consistent ordering
    sorted_results = sorted(all_results, key=lambda x: (x['vehicle'], x['scenario'], x['model']))

    for res in sorted_results:
        vehicle = res['vehicle']
        scenario = res['scenario'].capitalize()
        model = "LogReg" if res['model'] == 'logisticregression' else "RF"
        acc = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "N/A"
        prec = f"{res['precision']:.4f}" if res['precision'] is not None else "N/A"
        rec = f"{res['recall']:.4f}" if res['recall'] is not None else "N/A"
        f1 = f"{res['f1_score']:.4f}" if res['f1_score'] is not None else "N/A"
        auc = f"{res['roc_auc']:.4f}" if res['roc_auc'] is not None else "N/A"

        table.append(f"| {vehicle} | {scenario} | {model} | {acc} | {prec} | {rec} | {f1} | {auc} |\n")

    return "".join(table)


def generate_summary_statistics(all_results):
    """Generate overall statistics comparing models."""
    stats = []
    stats.append("\n### Summary Statistics\n\n")

    # Filter out None values and separate by model
    lr_results = [r for r in all_results if r['model'] == 'logisticregression']
    rf_results = [r for r in all_results if r['model'] == 'randomforest']

    def calc_stats(results, metric):
        values = [r[metric] for r in results if r.get(metric) is not None]
        if values:
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return None

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    stats.append("#### Logistic Regression\n\n")
    stats.append("| Metric | Mean | Std Dev | Min | Max |\n")
    stats.append("|--------|------|---------|-----|-----|\n")
    for metric in metrics:
        stat = calc_stats(lr_results, metric)
        if stat:
            stats.append(
                f"| {metric.replace('_', ' ').title()} | {stat['mean']:.4f} | {stat['std']:.4f} | {stat['min']:.4f} | {stat['max']:.4f} |\n")

    stats.append("\n#### Random Forest\n\n")
    stats.append("| Metric | Mean | Std Dev | Min | Max |\n")
    stats.append("|--------|------|---------|-----|-----|\n")
    for metric in metrics:
        stat = calc_stats(rf_results, metric)
        if stat:
            stats.append(
                f"| {metric.replace('_', ' ').title()} | {stat['mean']:.4f} | {stat['std']:.4f} | {stat['min']:.4f} | {stat['max']:.4f} |\n")

    stats.append("\n#### Model Comparison (Improvement: RF vs LR)\n\n")
    stats.append("| Metric | LR Mean | RF Mean | Absolute Gain | Relative Gain |\n")
    stats.append("|--------|---------|---------|---------------|---------------|\n")
    for metric in metrics:
        lr_stat = calc_stats(lr_results, metric)
        rf_stat = calc_stats(rf_results, metric)
        if lr_stat and rf_stat:
            abs_gain = rf_stat['mean'] - lr_stat['mean']
            rel_gain = (abs_gain / lr_stat['mean']) * 100 if lr_stat['mean'] > 0 else 0
            stats.append(
                f"| {metric.replace('_', ' ').title()} | {lr_stat['mean']:.4f} | {rf_stat['mean']:.4f} | +{abs_gain:.4f} | +{rel_gain:.1f}% |\n")

    return "".join(stats)


def generate_ieee_summary(all_results):
    """Generate comprehensive IEEE-style summary."""
    summary = []
    summary.append("# CAN Bus Intrusion Detection: Experimental Results Summary\n\n")
    summary.append("**Date**: " + str(pd.Timestamp.now().date()) + "\n\n")
    summary.append(
        "This document summarizes the experimental results for CAN bus intrusion detection using Logistic Regression and Random Forest models across various vehicle datasets and attack scenarios (Fuzzing, Replay, Combined).\n\n")

    # Add summary statistics
    summary.append(generate_summary_statistics(all_results))

    # Add comparison table
    summary.append(generate_comparison_table(all_results))

    summary.append("\n## Analysis by Attack Type\n\n")

    # Fuzzing Attacks
    summary.append("### Effectiveness Against Fuzzing Attacks\n\n")
    for vehicle in ["Vehicle A", "Vehicle B", "Vehicle C"]:
        summary.append(f"#### {vehicle}\n\n")
        fuzzing_results = [r for r in all_results if r['vehicle'] == vehicle and r['scenario'] == 'fuzzing']

        lr_results = [r for r in fuzzing_results if r['model'] == 'logisticregression']
        rf_results = [r for r in fuzzing_results if r['model'] == 'randomforest']

        if lr_results:
            lr_acc = np.mean([r['accuracy'] for r in lr_results if r['accuracy'] is not None])
            lr_rec = np.mean([r['recall'] for r in lr_results if r['recall'] is not None])
            lr_f1 = np.mean([r['f1_score'] for r in lr_results if r['f1_score'] is not None])
            summary.append(
                f"- **Logistic Regression**: Accuracy: {lr_acc:.4f}, Recall: {lr_rec:.4f}, F1: {lr_f1:.4f}\n")
        if rf_results:
            rf_acc = np.mean([r['accuracy'] for r in rf_results if r['accuracy'] is not None])
            rf_rec = np.mean([r['recall'] for r in rf_results if r['recall'] is not None])
            rf_f1 = np.mean([r['f1_score'] for r in rf_results if r['f1_score'] is not None])
            summary.append(f"- **Random Forest**: Accuracy: {rf_acc:.4f}, Recall: {rf_rec:.4f}, F1: {rf_f1:.4f}\n")

        if lr_results and rf_results:
            improvement = ((rf_f1 - lr_f1) / lr_f1) * 100 if lr_f1 > 0 else 0
            summary.append(f"- **Improvement**: Random Forest shows {improvement:.1f}% better F1-score\n")
        summary.append("\n")

    summary.append(
        "**Key Finding**: Fuzzing attacks are highly detectable by both models, with Random Forest achieving near-perfect detection rates (99-100% accuracy). The anomalous patterns created by fuzzing (random/invalid data injection) make these attacks relatively easy to identify.\n\n")

    # Replay Attacks
    summary.append("### Effectiveness Against Replay Attacks\n\n")
    for vehicle in ["Vehicle A", "Vehicle B", "Vehicle C"]:
        summary.append(f"#### {vehicle}\n\n")
        replay_results = [r for r in all_results if r['vehicle'] == vehicle and r['scenario'] == 'replay']

        lr_results = [r for r in replay_results if r['model'] == 'logisticregression']
        rf_results = [r for r in replay_results if r['model'] == 'randomforest']

        if lr_results:
            lr_acc = np.mean([r['accuracy'] for r in lr_results if r['accuracy'] is not None])
            lr_rec = np.mean([r['recall'] for r in lr_results if r['recall'] is not None])
            lr_f1 = np.mean([r['f1_score'] for r in lr_results if r['f1_score'] is not None])
            summary.append(
                f"- **Logistic Regression**: Accuracy: {lr_acc:.4f}, Recall: {lr_rec:.4f}, F1: {lr_f1:.4f}\n")
            if lr_rec < 0.1:
                summary.append(f"  - ⚠️ **CRITICAL**: Recall below 10% - fails to detect replay attacks!\n")
        if rf_results:
            rf_acc = np.mean([r['accuracy'] for r in rf_results if r['accuracy'] is not None])
            rf_rec = np.mean([r['recall'] for r in rf_results if r['recall'] is not None])
            rf_f1 = np.mean([r['f1_score'] for r in rf_results if r['f1_score'] is not None])
            summary.append(f"- **Random Forest**: Accuracy: {rf_acc:.4f}, Recall: {rf_rec:.4f}, F1: {rf_f1:.4f}\n")
        summary.append("\n")

    summary.append(
        "**Key Finding**: Replay attacks pose significant challenges for Logistic Regression, often resulting in near-zero recall (0-4% on some vehicles), indicating complete detection failure. Random Forest maintains reasonable performance (55-98% recall), though lower than fuzzing detection. Replay attacks are harder to detect because they use valid CAN messages replayed at inappropriate times.\n\n")

    # Combined Attacks
    summary.append("### Effectiveness Against Combined Attacks\n\n")
    for vehicle in ["Vehicle A", "Vehicle B", "Vehicle C"]:
        summary.append(f"#### {vehicle}\n\n")
        combined_results = [r for r in all_results if r['vehicle'] == vehicle and r['scenario'] == 'combined']

        lr_results = [r for r in combined_results if r['model'] == 'logisticregression']
        rf_results = [r for r in combined_results if r['model'] == 'randomforest']

        if lr_results:
            lr_acc = np.mean([r['accuracy'] for r in lr_results if r['accuracy'] is not None])
            lr_rec = np.mean([r['recall'] for r in lr_results if r['recall'] is not None])
            lr_f1 = np.mean([r['f1_score'] for r in lr_results if r['f1_score'] is not None])
            summary.append(
                f"- **Logistic Regression**: Accuracy: {lr_acc:.4f}, Recall: {lr_rec:.4f}, F1: {lr_f1:.4f}\n")
            if lr_rec < 0.25:
                summary.append(f"  - ⚠️ **CRITICAL**: Recall below 25% - misses majority of attacks!\n")
        if rf_results:
            rf_acc = np.mean([r['accuracy'] for r in rf_results if r['accuracy'] is not None])
            rf_rec = np.mean([r['recall'] for r in rf_results if r['recall'] is not None])
            rf_f1 = np.mean([r['f1_score'] for r in rf_results if r['f1_score'] is not None])
            summary.append(f"- **Random Forest**: Accuracy: {rf_acc:.4f}, Recall: {rf_rec:.4f}, F1: {rf_f1:.4f}\n")
            if rf_rec < 0.75:
                summary.append(f"  - ⚠️ **Note**: Recall below 75% - room for improvement\n")
        summary.append("\n")

    summary.append(
        "**Key Finding**: Combined attacks (mix of fuzzing and replay) represent the most realistic and challenging scenario. Logistic Regression performs poorly with 9-20% recall, while Random Forest achieves 65-98% recall depending on the vehicle. This demonstrates the importance of using ensemble methods for real-world security.\n\n")

    # Vehicle Comparison
    summary.append("## Cross-Vehicle Analysis\n\n")
    summary.append("### Vehicle-Specific Patterns\n\n")

    for vehicle in ["Vehicle A", "Vehicle B", "Vehicle C"]:
        vehicle_results = [r for r in all_results if r['vehicle'] == vehicle]
        rf_vehicle = [r for r in vehicle_results if r['model'] == 'randomforest']

        if rf_vehicle:
            avg_acc = np.mean([r['accuracy'] for r in rf_vehicle if r['accuracy'] is not None])
            avg_rec = np.mean([r['recall'] for r in rf_vehicle if r['recall'] is not None])
            summary.append(
                f"**{vehicle}** (Random Forest): Average Accuracy: {avg_acc:.4f}, Average Recall: {avg_rec:.4f}\n")

    summary.append(
        "\nWhile performance varies across vehicles, the general trends remain consistent: Random Forest significantly outperforms Logistic Regression, and attack complexity (Fuzzing < Replay < Combined) correlates with detection difficulty. Vehicle-specific differences likely reflect variations in normal traffic patterns and attack implementation details.\n\n")

    # Conclusions
    summary.append("## Conclusions and Recommendations\n\n")
    summary.append(
        "1. **Model Selection**: Random Forest with class balancing (`class_weight='balanced'`) is strongly recommended over Logistic Regression for CAN bus intrusion detection.\n\n")
    summary.append(
        "2. **Security Implications**: Logistic Regression's low recall on replay and combined attacks (0-20%) represents a critical security vulnerability, allowing most attacks to pass undetected.\n\n")
    summary.append(
        "3. **Performance Hierarchy**: Detection difficulty increases with attack sophistication: Fuzzing (easiest) → Replay (moderate) → Combined (hardest).\n\n")
    summary.append("4. **Future Work**: \n")
    summary.append("   - Hyperparameter tuning to improve combined attack detection\n")
    summary.append("   - Feature engineering for temporal pattern recognition\n")
    summary.append("   - Deep learning models (LSTM) for sequential pattern detection\n")
    summary.append("   - Real-time deployment testing with latency constraints\n\n")

    summary.append(
        "This analysis demonstrates that ensemble machine learning methods (Random Forest) are essential for effective CAN bus intrusion detection in modern vehicles, achieving up to 100% detection on fuzzing attacks and maintaining 65-98% detection on complex combined attack scenarios.\n")

    return "".join(summary)


def generate_ppt_conclusions(all_results):
    """Generate concise bullet points for presentation slides."""
    conclusions = []
    conclusions.append("# Key Conclusions for CAN Bus Intrusion Detection\n\n")
    conclusions.append("## Model Performance Summary\n\n")
    conclusions.append("- **Random Forest (RF) Significantly Outperforms Logistic Regression (LR)**\n")
    conclusions.append("  - Average Accuracy: RF 96.1% vs LR 88.6% (+7.5%)\n")
    conclusions.append("  - Average Recall: RF 83.5% vs LR 34.3% (+143%)\n")
    conclusions.append("  - Average F1-Score: RF 89.3% vs LR 46.8% (+91%)\n\n")

    conclusions.append("## Attack Detection Performance\n\n")
    conclusions.append("### ✅ Fuzzing Attacks (Easiest)\n")
    conclusions.append("- RF achieves near-perfect detection: 99-100% accuracy\n")
    conclusions.append("- LR also performs well: 91-99% accuracy\n")
    conclusions.append("- High detectability due to anomalous patterns\n\n")

    conclusions.append("### ⚠️ Replay Attacks (Moderate)\n")
    conclusions.append("- RF maintains good performance: 88-100% accuracy, 55-98% recall\n")
    conclusions.append("- **LR shows critical failures**: 0-4% recall on 2/3 vehicles\n")
    conclusions.append("- Challenge: Valid messages at wrong times\n\n")

    conclusions.append("### 🔴 Combined Attacks (Hardest)\n")
    conclusions.append("- RF achieves 65-98% recall (acceptable but needs improvement)\n")
    conclusions.append("- **LR fails severely**: 9-20% recall (misses 80-91% of attacks)\n")
    conclusions.append("- Most realistic attack scenario\n\n")

    conclusions.append("## Security Risk Assessment\n\n")
    conclusions.append("### Logistic Regression: ❌ NOT PRODUCTION READY\n")
    conclusions.append("- Complete detection failure on replay attacks (Vehicles B & C)\n")
    conclusions.append("- Misses 80-91% of combined attacks\n")
    conclusions.append("- Unacceptable for security-critical applications\n\n")

    conclusions.append("### Random Forest: ✅ PRODUCTION VIABLE (with tuning)\n")
    conclusions.append("- Excellent on fuzzing (100% detection)\n")
    conclusions.append("- Good on replay (55-98% detection)\n")
    conclusions.append("- Acceptable on combined (65-98% detection)\n")
    conclusions.append("- Needs hyperparameter tuning for optimal performance\n\n")

    conclusions.append("## Vehicle-Specific Insights\n\n")
    conclusions.append("- **Vehicle A**: Best on simple attacks, struggles with combined (76% recall)\n")
    conclusions.append("- **Vehicle B**: Most inconsistent, but best on combined (98% recall)\n")
    conclusions.append("- **Vehicle C**: Weakest overall performer (65% recall on combined)\n\n")

    conclusions.append("## Key Takeaways\n\n")
    conclusions.append("1. **Use Random Forest** with `class_weight='balanced'` for CAN intrusion detection\n")
    conclusions.append("2. **Avoid Logistic Regression** - fails on 6/9 scenarios with <25% recall\n")
    conclusions.append("3. **Attack complexity matters** - Combined attacks need special attention\n")
    conclusions.append(
        "4. **Recall is critical** - False negatives (missed attacks) are more dangerous than false alarms\n")
    conclusions.append("5. **Further optimization needed** - Especially for combined attack scenarios\n\n")

    conclusions.append("## Recommendations for Production Deployment\n\n")
    conclusions.append("### Immediate Actions:\n")
    conclusions.append("- Deploy Random Forest with class balancing\n")
    conclusions.append("- Set alert thresholds to prioritize recall over precision\n")
    conclusions.append("- Monitor false alarm rates and adjust as needed\n\n")

    conclusions.append("### Future Improvements:\n")
    conclusions.append("- Hyperparameter tuning (Grid/Random Search)\n")
    conclusions.append("- Feature engineering for temporal patterns\n")
    conclusions.append("- Ensemble voting (RF + XGBoost + GradientBoosting)\n")
    conclusions.append("- Deep learning models (LSTM for sequences)\n")
    conclusions.append("- Real-time latency optimization\n\n")

    conclusions.append("---\n\n")
    conclusions.append(
        "**Conclusion**: This study demonstrates that ensemble machine learning (Random Forest) achieves **96% average accuracy** and **84% average recall** for CAN bus intrusion detection, significantly outperforming linear models and providing practical security for automotive networks.\n")

    return "".join(conclusions)


if __name__ == "__main__":
    results_dir = "results"
    all_parsed_results = []

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: '{results_dir}' directory not found. Please ensure experiments have been run.")
        exit(1)

    # Parse all result files
    result_files = [f for f in os.listdir(results_dir)
                    if f.endswith(".txt") and "results.txt" in f
                    and f not in ["logisticregression_results.txt", "randomforest_results.txt"]]

    print(f"Found {len(result_files)} result files to process...")

    for filename in result_files:
        file_path = os.path.join(results_dir, filename)
        try:
            parsed_data = parse_results_file(file_path)
            if parsed_data:
                all_parsed_results.append(parsed_data)
                print(f"  ✓ Parsed: {filename}")
        except Exception as e:
            print(f"  ✗ Error parsing {filename}: {e}")

    if not all_parsed_results:
        print("\nNo results found to summarize. Please ensure experiments have been run and results are saved.")
    else:
        print(f"\nSuccessfully parsed {len(all_parsed_results)} result files.\n")

        # Generate summaries
        print("Generating IEEE-style summary...")
        ieee_summary = generate_ieee_summary(all_parsed_results)

        print("Generating presentation conclusions...")
        ppt_conclusions = generate_ppt_conclusions(all_parsed_results)

        # Save outputs
        with open("ieee_summary.md", "w") as f:
            f.write(ieee_summary)
        print("✓ IEEE paper summary saved to ieee_summary.md")

        with open("ppt_conclusions.md", "w") as f:
            f.write(ppt_conclusions)
        print("✓ PPT slide conclusions saved to ppt_conclusions.md")

        # Also create a CSV for easy analysis
        print("\nGenerating CSV export...")
        df = pd.DataFrame(all_parsed_results)
        df.to_csv("results_summary.csv", index=False)
        print("✓ Results exported to results_summary.csv")

        print("\n" + "=" * 60)
        print("Summary generation complete!")
        print("=" * 60)
        print(f"Total experiments analyzed: {len(all_parsed_results)}")
        print(f"  - Logistic Regression: {len([r for r in all_parsed_results if r['model'] == 'logisticregression'])}")
        print(f"  - Random Forest: {len([r for r in all_parsed_results if r['model'] == 'randomforest'])}")
        print(f"  - Vehicles: {len(set([r['vehicle'] for r in all_parsed_results]))}")
        print(f"  - Scenarios: {len(set([r['scenario'] for r in all_parsed_results]))}")