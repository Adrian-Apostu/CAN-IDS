import os
import re
import numpy as np
import pandas as pd


def detect_scenario_from_path(dataset_path):
    """
    Detect the attack scenario from the dataset file path.
    Reads the folder name (Fuzzing / Replay / Combined), NOT the filename,
    so files named '0x181' or 'all IDs' are classified correctly.
    """
    normalised = dataset_path.replace("\\", "/")
    if "/Combined/" in normalised:
        return "combined"
    elif "/Fuzzing/" in normalised:
        return "fuzzing"
    elif "/Replay/" in normalised:
        return "replay"
    return "no_attack"


def parse_results_file(file_path):
    """Parse a single results file and extract all metrics."""
    with open(file_path, 'r') as f:
        content = f.read()

    results = {}
    filename = os.path.basename(file_path)

    # ------------------------------------------------------------------ #
    # 1. Extract model name from filename
    # ------------------------------------------------------------------ #
    if "logisticregression" in filename:
        results['model'] = "logisticregression"
    elif "randomforest" in filename:
        results['model'] = "randomforest"
    elif "gradientboosting" in filename:
        results['model'] = "gradientboosting"
    else:
        results['model'] = "unknown"

    results['filename'] = filename

    # ------------------------------------------------------------------ #
    # 2. Extract the dataset path from INSIDE the file content.
    #    The analyzer writes a line like:  Dataset: ./Datasets with attacks/...
    #    This is the ground truth — use it for scenario AND vehicle detection.
    # ------------------------------------------------------------------ #
    dataset_match = re.search(r"Dataset:\s*(.+)", content)
    if dataset_match:
        dataset_path = dataset_match.group(1).strip()
        results['dataset'] = dataset_path

        # Vehicle
        if "Vehicle A" in dataset_path:
            results['vehicle'] = "Vehicle A"
        elif "Vehicle B" in dataset_path:
            results['vehicle'] = "Vehicle B"
        elif "Vehicle C" in dataset_path:
            results['vehicle'] = "Vehicle C"
        else:
            results['vehicle'] = "Unknown"

        # Scenario — use folder path, not filename keywords
        results['scenario'] = detect_scenario_from_path(dataset_path)

    else:
        # Fallback: try to reconstruct from filename (old behaviour)
        results['dataset'] = filename
        results['vehicle'] = "Unknown"
        results['scenario'] = "Unknown"

    # ------------------------------------------------------------------ #
    # 3. Extract numeric metrics
    # ------------------------------------------------------------------ #
    accuracy_match = re.search(r"Accuracy.*?:\s*([01]\.\d{4})", content)
    results['accuracy'] = float(accuracy_match.group(1)) if accuracy_match else None

    precision_match = re.search(r"Precision.*?:\s*([01]\.\d{4})", content)
    results['precision'] = float(precision_match.group(1)) if precision_match else None

    recall_match = re.search(r"Recall.*?:\s*([01]\.\d{4})", content)
    results['recall'] = float(recall_match.group(1)) if recall_match else None

    f1_match = re.search(r"F1-score.*?:\s*([01]\.\d{4})", content)
    results['f1_score'] = float(f1_match.group(1)) if f1_match else None

    roc_auc_match = re.search(r"ROC AUC.*?:\s*([01]\.\d{4})", content)
    results['roc_auc'] = float(roc_auc_match.group(1)) if roc_auc_match else None

    # Confusion Matrix
    cm_match = re.search(
        r"Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\]\]",
        content, re.MULTILINE
    )
    if cm_match:
        results['tn'] = int(cm_match.group(1))
        results['fp'] = int(cm_match.group(2))
        results['fn'] = int(cm_match.group(3))
        results['tp'] = int(cm_match.group(4))
    else:
        results['tn'] = results['fp'] = results['fn'] = results['tp'] = None

    return results


def generate_comparison_table(all_results):
    """Generate a detailed comparison table."""
    table = []
    table.append("\n### Detailed Performance Comparison Table\n\n")
    table.append("| Vehicle | Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |\n")
    table.append("|---------|----------|-------|----------|-----------|--------|----------|----------|\n")

    sorted_results = sorted(all_results, key=lambda x: (x['vehicle'], x['scenario'], x['model']))

    for res in sorted_results:
        vehicle = res['vehicle']
        scenario = res['scenario'].capitalize()
        model_labels = {
            'logisticregression': 'LogReg',
            'randomforest': 'RF',
            'gradientboosting': 'GB'
        }
        model = model_labels.get(res['model'], res['model'].upper())
        acc   = f"{res['accuracy']:.4f}"   if res['accuracy']   is not None else "N/A"
        prec  = f"{res['precision']:.4f}"  if res['precision']  is not None else "N/A"
        rec   = f"{res['recall']:.4f}"     if res['recall']     is not None else "N/A"
        f1    = f"{res['f1_score']:.4f}"   if res['f1_score']   is not None else "N/A"
        auc_v = f"{res['roc_auc']:.4f}"    if res['roc_auc']    is not None else "N/A"
        table.append(
            f"| {vehicle} | {scenario} | {model} | {acc} | {prec} | {rec} | {f1} | {auc_v} |\n"
        )

    return "".join(table)


def generate_summary_statistics(all_results):
    """Generate overall statistics comparing models."""
    stats = []
    stats.append("\n### Summary Statistics\n\n")

    lr_results = [r for r in all_results if r['model'] == 'logisticregression']
    rf_results = [r for r in all_results if r['model'] == 'randomforest']

    def calc_stats(results, metric):
        values = [r[metric] for r in results if r.get(metric) is not None]
        if values:
            return {
                'mean': np.mean(values),
                'std':  np.std(values),
                'min':  np.min(values),
                'max':  np.max(values)
            }
        return None

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    all_models = sorted(set(r['model'] for r in all_results if r['model'] != 'unknown'))
    model_display = {
        'logisticregression': 'Logistic Regression',
        'randomforest': 'Random Forest',
        'gradientboosting': 'Gradient Boosting'
    }

    for model_key in all_models:
        group = [r for r in all_results if r['model'] == model_key]
        label = model_display.get(model_key, model_key.title())
        stats.append(f"#### {label}\n\n")
        stats.append("| Metric | Mean | Std Dev | Min | Max |\n")
        stats.append("|--------|------|---------|-----|-----|\n")
        for metric in metrics:
            s = calc_stats(group, metric)
            if s:
                stats.append(
                    f"| {metric.replace('_', ' ').title()} "
                    f"| {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |\n"
                )
        stats.append("\n")

    stats.append("#### Model Comparison\n\n")
    stats.append("| Metric | " + " | ".join(model_display.get(m, m) for m in all_models) + " |\n")
    stats.append("|--------|" + "|".join(["------"] * len(all_models)) + "|\n")
    for metric in metrics:
        row = f"| {metric.replace('_', ' ').title()} |"
        for model_key in all_models:
            group = [r for r in all_results if r['model'] == model_key]
            s = calc_stats(group, metric)
            row += f" {s['mean']:.4f} |" if s else " N/A |"
        stats.append(row + "\n")

    return "".join(stats)


def generate_ieee_summary(all_results):
    """Generate comprehensive IEEE-style summary."""
    summary = []
    summary.append("# CAN Bus Intrusion Detection: Experimental Results Summary\n\n")
    summary.append("**Date**: " + str(pd.Timestamp.now().date()) + "\n\n")
    summary.append(
        "This document summarizes the experimental results for CAN bus intrusion detection "
        "using Logistic Regression and Random Forest models across various vehicle datasets "
        "and attack scenarios (Fuzzing, Replay, Combined).\n\n"
    )

    summary.append(generate_summary_statistics(all_results))
    summary.append(generate_comparison_table(all_results))
    summary.append("\n## Analysis by Attack Type\n\n")

    for attack_type, label in [("fuzzing", "Fuzzing"), ("replay", "Replay"), ("combined", "Combined")]:
        summary.append(f"### Effectiveness Against {label} Attacks\n\n")
        for vehicle in ["Vehicle A", "Vehicle B", "Vehicle C"]:
            summary.append(f"#### {vehicle}\n\n")
            type_results = [r for r in all_results
                            if r['vehicle'] == vehicle and r['scenario'] == attack_type]

            all_models_here = sorted(set(r['model'] for r in type_results if r['model'] != 'unknown'))
            model_display_local = {
                'logisticregression': 'Logistic Regression',
                'randomforest': 'Random Forest',
                'gradientboosting': 'Gradient Boosting'
            }

            for model_key in all_models_here:
                m_r = [r for r in type_results if r['model'] == model_key]
                m_label = model_display_local.get(model_key, model_key.title())
                if m_r:
                    m_acc = np.mean([r['accuracy']  for r in m_r if r['accuracy']  is not None])
                    m_rec = np.mean([r['recall']     for r in m_r if r['recall']    is not None])
                    m_f1  = np.mean([r['f1_score']   for r in m_r if r['f1_score']  is not None])
                    summary.append(
                        f"- **{m_label}**: Accuracy: {m_acc:.4f}, "
                        f"Recall: {m_rec:.4f}, F1: {m_f1:.4f}\n"
                    )
                    if m_rec < 0.25:
                        summary.append(
                            f"  - ⚠️ **CRITICAL**: Recall below 25% — misses majority of attacks!\n"
                        )
                    elif m_rec < 0.75:
                        summary.append(
                            f"  - ⚠️ **Note**: Recall below 75% — room for improvement\n"
                        )
                else:
                    summary.append(f"- **{m_label}**: No results found\n")

            summary.append("\n")

    # Cross-vehicle analysis
    summary.append("## Cross-Vehicle Analysis\n\n")
    summary.append("### Vehicle-Specific Patterns\n\n")
    for vehicle in ["Vehicle A", "Vehicle B", "Vehicle C"]:
        rf_v = [r for r in all_results
                if r['vehicle'] == vehicle and r['model'] == 'randomforest']
        if rf_v:
            avg_acc = np.mean([r['accuracy'] for r in rf_v if r['accuracy'] is not None])
            avg_rec = np.mean([r['recall']   for r in rf_v if r['recall']   is not None])
            summary.append(
                f"**{vehicle}** (Random Forest): "
                f"Average Accuracy: {avg_acc:.4f}, Average Recall: {avg_rec:.4f}\n"
            )

    summary.append(
        "\nWhile performance varies across vehicles, the general trends remain consistent: "
        "Random Forest significantly outperforms Logistic Regression, and attack complexity "
        "(Fuzzing < Replay < Combined) correlates with detection difficulty.\n\n"
    )

    # Conclusions
    summary.append("## Conclusions and Recommendations\n\n")
    summary.append(
        "1. **Model Selection**: Random Forest with class balancing (`class_weight='balanced'`) "
        "is strongly recommended over Logistic Regression for CAN bus intrusion detection.\n\n"
    )
    summary.append(
        "2. **Security Implications**: Logistic Regression's low recall on replay and combined "
        "attacks (0-20%) represents a critical security vulnerability, allowing most attacks to "
        "pass undetected.\n\n"
    )
    summary.append(
        "3. **Performance Hierarchy**: Detection difficulty increases with attack sophistication: "
        "Fuzzing (easiest) → Replay (moderate) → Combined (hardest).\n\n"
    )
    summary.append("4. **Future Work**: \n")
    summary.append("   - Hyperparameter tuning to improve combined attack detection\n")
    summary.append("   - Feature engineering for temporal pattern recognition\n")
    summary.append("   - Deep learning models (LSTM) for sequential pattern detection\n")
    summary.append("   - Real-time deployment testing with latency constraints\n\n")
    summary.append(
        "This analysis demonstrates that ensemble machine learning methods (Random Forest) are "
        "essential for effective CAN bus intrusion detection in modern vehicles.\n"
    )

    return "".join(summary)


def generate_ppt_conclusions(all_results):
    """Generate concise bullet points for presentation slides."""
    conclusions = []
    conclusions.append("# Key Conclusions for CAN Bus Intrusion Detection\n\n")

    def mean_metric(results, metric):
        vals = [r[metric] for r in results if r.get(metric) is not None]
        return np.mean(vals) * 100 if vals else 0

    all_models = sorted(set(r['model'] for r in all_results if r['model'] != 'unknown'))
    model_display = {
        'logisticregression': 'Logistic Regression',
        'randomforest': 'Random Forest',
        'gradientboosting': 'Gradient Boosting'
    }

    # Build per-model stats
    model_stats = {}
    for model_key in all_models:
        group = [r for r in all_results if r['model'] == model_key]
        model_stats[model_key] = {
            'acc': mean_metric(group, 'accuracy'),
            'rec': mean_metric(group, 'recall'),
            'f1':  mean_metric(group, 'f1_score'),
            'label': model_display.get(model_key, model_key.title())
        }

    conclusions.append("## Model Performance Summary\n\n")
    for model_key in all_models:
        s = model_stats[model_key]
        conclusions.append(
            f"- **{s['label']}**: "
            f"Accuracy {s['acc']:.1f}%, Recall {s['rec']:.1f}%, F1 {s['f1']:.1f}%\n"
        )
    conclusions.append("\n")

    conclusions.append("## Attack Detection Performance\n\n")
    conclusions.append("### ✅ Fuzzing Attacks (Easiest)\n")
    conclusions.append("- High detectability due to anomalous random/invalid data patterns\n\n")
    conclusions.append("### ⚠️ Replay Attacks (Moderate)\n")
    conclusions.append("- Challenge: Valid messages injected at wrong times\n\n")
    conclusions.append("### 🔴 Combined Attacks (Hardest)\n")
    conclusions.append("- Most realistic attack scenario — mix of replay and fuzzing\n\n")

    conclusions.append("## Key Takeaways\n\n")
    conclusions.append("1. **Use ensemble methods** (Random Forest / Gradient Boosting) "
                       "with `class_weight='balanced'`\n")
    conclusions.append("2. **Avoid Logistic Regression** — fails on replay and combined attacks\n")
    conclusions.append("3. **Recall is critical** — missed attacks are more dangerous than false alarms\n")
    conclusions.append("4. **Attack complexity matters** — combined attacks need special attention\n\n")

    conclusions.append("---\n\n")
    # Best model = highest recall
    best_key = max(all_models, key=lambda m: model_stats[m]['rec'])
    worst_key = min(all_models, key=lambda m: model_stats[m]['rec'])
    best = model_stats[best_key]
    worst = model_stats[worst_key]
    conclusions.append(
        f"**Conclusion**: {best['label']} achieves {best['acc']:.0f}% average accuracy and "
        f"{best['rec']:.0f}% average recall, significantly outperforming "
        f"{worst['label']} ({worst['acc']:.0f}% accuracy, {worst['rec']:.0f}% recall).\n"
    )

    return "".join(conclusions)


if __name__ == "__main__":
    results_dir = "results"
    all_parsed_results = []

    if not os.path.exists(results_dir):
        print(f"Error: '{results_dir}' directory not found. Please run experiments first.")
        exit(1)

    result_files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".txt") and "results.txt" in f
        and f not in ["logisticregression_results.txt", "randomforest_results.txt"]
    ]

    print(f"Found {len(result_files)} result files to process...")

    for filename in result_files:
        file_path = os.path.join(results_dir, filename)
        try:
            parsed_data = parse_results_file(file_path)
            if parsed_data:
                all_parsed_results.append(parsed_data)
                print(f"  ✓ Parsed: {filename}  →  vehicle={parsed_data['vehicle']}"
                      f"  scenario={parsed_data['scenario']}  model={parsed_data['model']}")
        except Exception as e:
            print(f"  ✗ Error parsing {filename}: {e}")

    if not all_parsed_results:
        print("\nNo results found. Please ensure experiments have been run.")
    else:
        print(f"\nSuccessfully parsed {len(all_parsed_results)} result files.\n")

        print("Generating IEEE-style summary...")
        ieee_summary = generate_ieee_summary(all_parsed_results)

        print("Generating presentation conclusions...")
        ppt_conclusions = generate_ppt_conclusions(all_parsed_results)

        with open("ieee_summary.md", "w") as f:
            f.write(ieee_summary)
        print("✓ IEEE paper summary saved to ieee_summary.md")

        with open("ppt_conclusions.md", "w") as f:
            f.write(ppt_conclusions)
        print("✓ PPT slide conclusions saved to ppt_conclusions.md")

        df = pd.DataFrame(all_parsed_results)
        df.to_csv("results_summary.csv", index=False)
        print("✓ Results exported to results_summary.csv")

        print("\n" + "=" * 60)
        print("Summary generation complete!")
        print("=" * 60)
        print(f"Total experiments analysed: {len(all_parsed_results)}")
        print(f"  - Logistic Regression : {len([r for r in all_parsed_results if r['model'] == 'logisticregression'])}")
        print(f"  - Random Forest       : {len([r for r in all_parsed_results if r['model'] == 'randomforest'])}")
        print(f"  - Vehicles            : {sorted(set(r['vehicle']  for r in all_parsed_results))}")
        print(f"  - Scenarios           : {sorted(set(r['scenario'] for r in all_parsed_results))}")