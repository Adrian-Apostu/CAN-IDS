import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, auc, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


def load_and_preprocess_data(file_path, sample_frac=None, random_state=42):
    """Loads a CSV file, preprocesses it, and returns UNSCALED features + labels.

    FIXED (Step 1.1): No longer applies StandardScaler here. Scaling must happen
    AFTER train/test split to prevent data leakage.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep=';', header=None)

    print(f"Original dataset shape: {df.shape}")

    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"Sampled {sample_frac * 100}% of the dataset. New shape: {df.shape}")

    # Last column is the class label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert all feature columns to numeric, coercing errors
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # Check class distribution
    print("\nClass distribution:")
    print(y.value_counts())
    print(f"Class balance: {y.value_counts(normalize=True)}")

    # Encode labels if they're strings
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"\nEncoded labels: {dict(enumerate(label_encoder.classes_))}")

    # Determine if binary or multiclass
    n_classes = len(np.unique(y))
    print(f"\nNumber of classes: {n_classes}")
    is_binary = n_classes == 2

    print("Data loaded and preprocessed (unscaled).")
    return X.values, y.values, is_binary, label_encoder


def scale_features(X_train, X_test):
    """Fit StandardScaler on training data only, then transform both sets.

    This prevents data leakage: the scaler learns statistics (mean, std)
    only from the training set, so the test set remains truly unseen.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def create_model(model_name):
    """Create a fresh model instance for each fold.

    Returns (model, needs_sample_weight) tuple.
    """
    if model_name == "LogisticRegression":
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced'  # FIXED (Step 1.2)
        )
        return model, False
    elif model_name == "SVM":
        model = SVC(random_state=42, probability=True)
        return model, False
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced'
        )
        return model, False
    elif model_name == "GradientBoosting":
        # NOTE (Step 1.3): Balanced sample_weight was tested but caused
        # regression on datasets where GB already performed well.
        # GB is kept unweighted as a high-precision comparison point.
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        return model, False
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_fold(model, X_train, y_train, X_test, y_test, is_binary,
                  sample_weight=None):
    """Train and evaluate a single fold. Returns a dict of metrics."""

    # Scale features inside the fold (Step 1.1: no leakage)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train
    if sample_weight is not None:
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    avg_method = 'binary' if is_binary else 'weighted'
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg_method, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # ROC AUC
    roc_auc_val = None
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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc_val,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
    }


def run_kfold_cv(X, y, model_name, is_binary, n_splits=5, random_state=42):
    """Run Stratified K-Fold cross-validation.

    Step 1.4: Replaces single train/test split with K-Fold CV.
    Each fold gets its own scaler (fit on that fold's training data only).

    Returns:
        fold_results: list of per-fold metric dicts
        aggregate: dict with mean, std for each metric + summed confusion matrix
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    all_y_test = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create a fresh model instance for each fold
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
              f"{roc_str}")

    # Aggregate results
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    aggregate = {}
    for key in metrics_keys:
        values = [f[key] for f in fold_results if f[key] is not None]
        if values:
            aggregate[f'{key}_mean'] = np.mean(values)
            aggregate[f'{key}_std'] = np.std(values)
            aggregate[f'{key}_values'] = values

    # Summed confusion matrix across all folds
    aggregate['confusion_matrix_sum'] = sum(f['confusion_matrix'] for f in fold_results)

    # Full classification report on aggregated predictions
    aggregate['y_test_all'] = np.array(all_y_test)
    aggregate['y_pred_all'] = np.array(all_y_pred)

    return fold_results, aggregate


def save_results(model_name, dataset_name, is_binary, fold_results, aggregate,
                 n_splits, results_dir="results"):
    """Save results in backward-compatible format + new CV statistics.

    The output format preserves the same lines the summarizer expects:
        Accuracy: 0.XXXX          (mean across folds)
        Precision (binary): 0.XXXX
        Recall (binary): 0.XXXX
        F1-score (binary): 0.XXXX
        Confusion Matrix: [[...]]
        ROC AUC (binary): 0.XXXX
    Plus new lines with +/- std and per-fold details.
    """

    avg_method = 'binary' if is_binary else 'weighted'

    acc_mean = aggregate['accuracy_mean']
    prec_mean = aggregate['precision_mean']
    rec_mean = aggregate['recall_mean']
    f1_mean = aggregate['f1_score_mean']
    acc_std = aggregate['accuracy_std']
    prec_std = aggregate['precision_std']
    rec_std = aggregate['recall_std']
    f1_std = aggregate['f1_score_std']

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
            f"AUC={roc_str}\n"
        )

    print(results_summary)

    # Save to file
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(dataset_name)))
    sanitized_dataset_name = (os.path.basename(dataset_name)
                              .replace('.csv', '')
                              .replace(' ', '_')
                              .replace('-', '_')
                              .replace('.', '_'))
    file_name = f"{parent_folder}___{sanitized_dataset_name}_{model_name.lower().replace(' ', '_')}_results.txt"

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, file_name), "w") as f:
        f.write(results_summary)
    print(f"Results saved to {os.path.join(results_dir, file_name)}")

    # ROC AUC line (backward compatible with summarizer)
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

    # Step 1: Load data (returns UNSCALED features)
    X, y, is_binary, label_encoder = load_and_preprocess_data(
        args.dataset, sample_frac=args.sample_frac
    )

    # Step 2 (Step 1.4): Run Stratified K-Fold Cross-Validation
    print(f"\nRunning {args.n_folds}-Fold Stratified Cross-Validation with {args.model}...")
    fold_results, aggregate = run_kfold_cv(
        X, y, args.model, is_binary, n_splits=args.n_folds
    )

    # Step 3: Save results (backward-compatible format)
    save_results(
        args.model, args.dataset, is_binary,
        fold_results, aggregate, args.n_folds,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()