import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, auc, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================================
# Feature names for CAN bus dataset (14 columns: 5 CAN ID window + 8 data + 1 label)
# ============================================================================
FEATURE_NAMES = [
    'CAN_ID_t', 'CAN_ID_t-1', 'CAN_ID_t-2', 'CAN_ID_t-3', 'CAN_ID_t-4',
    'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7'
]


def load_and_preprocess_data(file_path, sample_frac=None, random_state=42):
    """Loads a CSV file, preprocesses it, and returns UNSCALED features + labels."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep=';', header=None)
    print(f"Original dataset shape: {df.shape}")

    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"Sampled {sample_frac * 100}% of the dataset. New shape: {df.shape}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    print("\nClass distribution:")
    print(y.value_counts())
    print(f"Class balance: {y.value_counts(normalize=True)}")

    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"\nEncoded labels: {dict(enumerate(label_encoder.classes_))}")

    n_classes = len(np.unique(y))
    print(f"\nNumber of classes: {n_classes}")
    is_binary = n_classes == 2

    print("Data loaded and preprocessed (unscaled).")
    return X.values, y.values, is_binary, label_encoder


def scale_features(X_train, X_test):
    """Fit StandardScaler on training data only, then transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def create_model(model_name):
    """Create a fresh model instance for each fold."""
    if model_name == "LogisticRegression":
        return LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs',
                                  class_weight='balanced'), False
    elif model_name == "SVM":
        return SVC(random_state=42, probability=True), False
    elif model_name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_estimators=100,
                                      class_weight='balanced'), False
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42, n_estimators=100), False
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_fold(model, X_train, y_train, X_test, y_test, is_binary,
                  sample_weight=None):
    """Train and evaluate a single fold. Returns metrics + trained model artifacts."""

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train with timing
    t_start = time.time()
    if sample_weight is not None:
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train_scaled, y_train)
    train_time = time.time() - t_start

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    avg_method = 'binary' if is_binary else 'weighted'
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg_method, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # ROC data
    roc_auc_val = None
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_scaled)
            if is_binary:
                roc_auc_val = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc_val = roc_auc_score(y_test, y_proba,
                                            multi_class='ovr', average='macro')
        except Exception:
            pass

    # Feature importance (tree-based and LR)
    feat_importance = None
    if hasattr(model, 'feature_importances_'):
        feat_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For LR: use absolute coefficient values as importance proxy
        feat_importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 \
            else np.abs(model.coef_[0])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc_val,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'train_time': train_time,
        'feature_importance': feat_importance,
    }


def run_kfold_cv(X, y, model_name, is_binary, n_splits=5, random_state=42):
    """Run Stratified K-Fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    all_y_test = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, needs_sw = create_model(model_name)
        sample_weight = None
        if needs_sw:
            sample_weight = compute_sample_weight('balanced', y_train)

        fold_metrics = evaluate_fold(
            model, X_train, y_train, X_test, y_test,
            is_binary, sample_weight
        )

        fold_results.append(fold_metrics)
        all_y_test.extend(y_test)
        all_y_pred.extend(fold_metrics['y_pred'])

        roc_str = f"AUC={fold_metrics['roc_auc']:.4f}" if fold_metrics['roc_auc'] is not None else ""
        print(f"  Fold {fold_idx}/{n_splits}: "
              f"Acc={fold_metrics['accuracy']:.4f}  "
              f"Prec={fold_metrics['precision']:.4f}  "
              f"Rec={fold_metrics['recall']:.4f}  "
              f"F1={fold_metrics['f1_score']:.4f}  "
              f"{roc_str}  "
              f"Time={fold_metrics['train_time']:.1f}s")

    # Aggregate
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    aggregate = {}
    for key in metrics_keys:
        values = [f[key] for f in fold_results if f[key] is not None]
        if values:
            aggregate[f'{key}_mean'] = np.mean(values)
            aggregate[f'{key}_std'] = np.std(values)
            aggregate[f'{key}_values'] = values

    aggregate['confusion_matrix_sum'] = sum(f['confusion_matrix'] for f in fold_results)
    aggregate['y_test_all'] = np.array(all_y_test)
    aggregate['y_pred_all'] = np.array(all_y_pred)

    # Aggregate training time
    train_times = [f['train_time'] for f in fold_results]
    aggregate['train_time_mean'] = np.mean(train_times)
    aggregate['train_time_std'] = np.std(train_times)
    aggregate['train_time_total'] = np.sum(train_times)

    # Average feature importance across folds
    importances = [f['feature_importance'] for f in fold_results
                   if f['feature_importance'] is not None]
    if importances:
        aggregate['feature_importance_mean'] = np.mean(importances, axis=0)
        aggregate['feature_importance_std'] = np.std(importances, axis=0)

    return fold_results, aggregate


# ============================================================================
# Visualization functions (Phase 2)
# ============================================================================

def plot_confusion_matrix(cm, model_name, dataset_label, save_path):
    """Generate a confusion matrix heatmap and save as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Normal', 'Attack']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}\n{dataset_label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")


def plot_roc_curve(fold_results, model_name, dataset_label, is_binary, save_path):
    """Plot ROC curves for all folds + mean ROC."""
    if not is_binary:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i, fold in enumerate(fold_results):
        if fold['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(fold['y_test'], fold['y_proba'][:, 1])
            fold_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.3, linewidth=1,
                    label=f'Fold {i+1} (AUC={fold_auc:.3f})')
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([auc(mean_fpr, t) for t in tprs])
        ax.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
                label=f'Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue',
                        alpha=0.1, label='± 1 std. dev.')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {model_name}\n{dataset_label}', fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC curve saved: {save_path}")


def plot_feature_importance(importance_mean, importance_std, model_name,
                            dataset_label, save_path):
    """Plot feature importance bar chart."""
    if importance_mean is None:
        return

    n_features = len(importance_mean)
    names = FEATURE_NAMES[:n_features] if n_features <= len(FEATURE_NAMES) \
        else [f'Feature_{i}' for i in range(n_features)]

    # Sort by importance
    sorted_idx = np.argsort(importance_mean)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_imp = importance_mean[sorted_idx]
    sorted_std = importance_std[sorted_idx] if importance_std is not None else None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color: CAN ID features vs Data byte features
    colors = []
    for name in sorted_names:
        if name.startswith('CAN_ID'):
            colors.append('#2196F3')  # Blue for CAN ID
        else:
            colors.append('#FF9800')  # Orange for Data bytes

    bars = ax.barh(range(n_features), sorted_imp, color=colors, edgecolor='white')
    if sorted_std is not None:
        ax.errorbar(sorted_imp, range(n_features), xerr=sorted_std,
                    fmt='none', color='black', capsize=3, linewidth=1)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance — {model_name}\n{dataset_label}', fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='CAN ID (sliding window)'),
                       Patch(facecolor='#FF9800', label='Data Field (bytes)')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Feature importance saved: {save_path}")


# ============================================================================
# Results saving
# ============================================================================

def make_file_prefix(dataset_name, model_name):
    """Generate a consistent file prefix for results and figures."""
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(dataset_name)))
    sanitized = (os.path.basename(dataset_name)
                 .replace('.csv', '').replace(' ', '_')
                 .replace('-', '_').replace('.', '_'))
    model_slug = model_name.lower().replace(' ', '_')
    return f"{parent_folder}___{sanitized}_{model_slug}"


def make_dataset_label(dataset_name):
    """Extract a human-readable label from the dataset path."""
    base = os.path.basename(dataset_name).replace('.csv', '').replace('_', ' ')
    parent = os.path.basename(os.path.dirname(os.path.dirname(dataset_name)))
    if parent:
        return f"{parent} / {base}"
    return base


def save_results(model_name, dataset_name, is_binary, fold_results, aggregate,
                 n_splits, results_dir="results"):
    """Save text results + generate all figures."""

    avg_method = 'binary' if is_binary else 'weighted'
    file_prefix = make_file_prefix(dataset_name, model_name)
    dataset_label = make_dataset_label(dataset_name)

    acc_mean = aggregate['accuracy_mean']
    prec_mean = aggregate['precision_mean']
    rec_mean = aggregate['recall_mean']
    f1_mean = aggregate['f1_score_mean']
    acc_std = aggregate['accuracy_std']
    prec_std = aggregate['precision_std']
    rec_std = aggregate['recall_std']
    f1_std = aggregate['f1_score_std']
    train_mean = aggregate['train_time_mean']
    train_std = aggregate['train_time_std']
    train_total = aggregate['train_time_total']

    cm_sum = aggregate['confusion_matrix_sum']

    results_summary = f"""--- {model_name} Results ---
Dataset: {dataset_name}
Classification Type: {'Binary' if is_binary else 'Multiclass'}
Cross-Validation: {n_splits}-Fold Stratified
Accuracy: {acc_mean:.4f}
Accuracy Std: {acc_std:.4f}
Precision ({avg_method}): {prec_mean:.4f}
Precision Std: {prec_std:.4f}
Recall ({avg_method}): {rec_mean:.4f}
Recall Std: {rec_std:.4f}
F1-score ({avg_method}): {f1_mean:.4f}
F1-score Std: {f1_std:.4f}
Training Time (per fold): {train_mean:.2f}s +/- {train_std:.2f}s
Training Time (total CV): {train_total:.2f}s

Confusion Matrix:
{cm_sum}

Cross-Validation Note: Confusion matrix is summed across {n_splits} folds.

Detailed Classification Report (aggregated predictions):
{classification_report(aggregate['y_test_all'], aggregate['y_pred_all'], zero_division=0)}

Per-Fold Results:
"""

    for i, fold in enumerate(fold_results, 1):
        roc_str = f"{fold['roc_auc']:.4f}" if fold['roc_auc'] is not None else "N/A"
        results_summary += (
            f"  Fold {i}: Acc={fold['accuracy']:.4f}  "
            f"Prec={fold['precision']:.4f}  "
            f"Rec={fold['recall']:.4f}  "
            f"F1={fold['f1_score']:.4f}  "
            f"AUC={roc_str}  "
            f"Time={fold['train_time']:.2f}s\n"
        )

    # Feature importance summary
    fi_mean = aggregate.get('feature_importance_mean')
    if fi_mean is not None:
        results_summary += "\nFeature Importance (mean across folds):\n"
        n = len(fi_mean)
        names = FEATURE_NAMES[:n] if n <= len(FEATURE_NAMES) else [f'F{i}' for i in range(n)]
        sorted_idx = np.argsort(fi_mean)[::-1]
        for idx in sorted_idx:
            fi_std = aggregate.get('feature_importance_std')
            std_str = f" +/- {fi_std[idx]:.4f}" if fi_std is not None else ""
            results_summary += f"  {names[idx]:15s}: {fi_mean[idx]:.4f}{std_str}\n"

    print(results_summary)

    # Save text results
    os.makedirs(results_dir, exist_ok=True)
    file_name = f"{file_prefix}_results.txt"
    with open(os.path.join(results_dir, file_name), "w") as f:
        f.write(results_summary)
    print(f"Results saved to {os.path.join(results_dir, file_name)}")

    # ROC AUC line (backward compatible)
    roc_mean = aggregate.get('roc_auc_mean')
    roc_std = aggregate.get('roc_auc_std')
    if roc_mean is not None:
        if is_binary:
            roc_summary = f"\nROC AUC (binary): {roc_mean:.4f}\nROC AUC Std: {roc_std:.4f}\n"
        else:
            roc_summary = f"\nROC AUC (multiclass, macro-avg): {roc_mean:.4f}\nROC AUC Std: {roc_std:.4f}\n"
        print(roc_summary)
        with open(os.path.join(results_dir, file_name), "a") as f:
            f.write(roc_summary)

    # ---- Generate figures ----
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Confusion matrix heatmap
    plot_confusion_matrix(
        cm_sum, model_name, dataset_label,
        os.path.join(figures_dir, f"{file_prefix}_confusion_matrix.png")
    )

    # 2. ROC curve
    if is_binary:
        plot_roc_curve(
            fold_results, model_name, dataset_label, is_binary,
            os.path.join(figures_dir, f"{file_prefix}_roc_curve.png")
        )

    # 3. Feature importance
    fi_mean = aggregate.get('feature_importance_mean')
    fi_std = aggregate.get('feature_importance_std')
    if fi_mean is not None:
        plot_feature_importance(
            fi_mean, fi_std, model_name, dataset_label,
            os.path.join(figures_dir, f"{file_prefix}_feature_importance.png")
        )


def main():
    parser = argparse.ArgumentParser(description="CAN Bus Intrusion Detection Analyzer")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the CSV dataset file")
    parser.add_argument("--model", type=str, default="LogisticRegression",
                        choices=["LogisticRegression", "SVM", "RandomForest", "GradientBoosting"],
                        help="Machine learning model to use")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction of the dataset to sample (e.g., 0.1 for 10%%)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")

    args = parser.parse_args()

    X, y, is_binary, label_encoder = load_and_preprocess_data(
        args.dataset, sample_frac=args.sample_frac
    )

    print(f"\nRunning {args.n_folds}-Fold Stratified Cross-Validation with {args.model}...")
    fold_results, aggregate = run_kfold_cv(
        X, y, args.model, is_binary, n_splits=args.n_folds
    )

    save_results(
        args.model, args.dataset, is_binary,
        fold_results, aggregate, args.n_folds,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()