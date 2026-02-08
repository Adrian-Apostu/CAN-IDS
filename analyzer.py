import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, auc, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


def load_and_preprocess_data(file_path, sample_frac=None, random_state=42):
    '''Loads a CSV file, preprocesses it, and splits into features and labels.'''
    print(f"Loading data from {file_path}...")
    # Assuming semicolon as separator and no header
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

    # Fill any NaN values that might result from coercion with 0 or a suitable strategy
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

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data loaded and preprocessed.")

    return X_scaled, y, is_binary, label_encoder


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, model_name,
                             dataset_name, is_binary, results_dir="results"):
    '''Trains a given model, evaluates it, and prints/saves metrics.'''
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Use appropriate averaging for binary vs multiclass
    avg_method = 'binary' if is_binary else 'weighted'
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg_method, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    results_summary = f'''--- {model_name} Results ---
Dataset: {dataset_name}
Classification Type: {'Binary' if is_binary else 'Multiclass'}
Accuracy: {accuracy:.4f}
Precision ({avg_method}): {precision:.4f}
Recall ({avg_method}): {recall:.4f}
F1-score ({avg_method}): {f1_score:.4f}

Confusion Matrix:
{cm}

Detailed Classification Report:
{classification_report(y_test, y_pred, zero_division=0)}
'''
    print(results_summary)

    # Sanitize dataset_name for filename
    sanitized_dataset_name = os.path.basename(dataset_name).replace('.csv', '').replace(' ', '_').replace('-',
                                                                                                          '_').replace(
        '.', '_')
    file_name = f"{sanitized_dataset_name}_{model_name.lower().replace(' ', '_')}_results.txt"

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, file_name), "w") as f:
        f.write(results_summary)
    print(f"Results saved to {os.path.join(results_dir, file_name)}")

    # ROC Curve and AUC
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)

            if is_binary:
                # Binary classification ROC
                y_proba_pos = y_proba[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
                roc_auc = auc(fpr, tpr)
                roc_summary = f'\nROC AUC (binary): {roc_auc:.4f}\n'
            else:
                # Multiclass ROC - use macro average
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                roc_summary = f'\nROC AUC (multiclass, macro-avg): {roc_auc:.4f}\n'

            print(roc_summary)
            with open(os.path.join(results_dir, file_name), "a") as f:
                f.write(roc_summary)
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")

    return accuracy, precision, recall, f1_score


def main():
    parser = argparse.ArgumentParser(description="CAN Bus Intrusion Detection Analyzer")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--model", type=str, default="LogisticRegression",
                        choices=["LogisticRegression", "SVM", "RandomForest", "GradientBoosting"],
                        help="Machine learning model to use")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction of the dataset to sample (e.g., 0.1 for 10%%)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")

    args = parser.parse_args()

    X, y, is_binary, label_encoder = load_and_preprocess_data(
        args.dataset, sample_frac=args.sample_frac
    )

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model selection with appropriate parameters
    # FIXED: Removed multi_class parameter for compatibility
    if args.model == "LogisticRegression":
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
    elif args.model == "SVM":
        # probability=True for ROC AUC, but slow for large datasets
        model = SVC(random_state=42, probability=True)
    elif args.model == "RandomForest":
        # Use class_weight='balanced' to handle imbalance
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced'  # Important for imbalanced data!
        )
    elif args.model == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model, args.model,
        args.dataset, is_binary, results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()