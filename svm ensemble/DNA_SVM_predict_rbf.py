"""
DNA SVM Prediction Pipeline using RBF Kernel

Pipeline steps:
1. Train SVM classifiers with calibrated 10-fold CV evaluation
2. Generate predictions using ensemble averaging
3. Select top 1000 sequences per color class based on minimum scores
4. Identify sequences by their indices from a key file

Usage:
    python DNA_SVM_predict_rbf.py
"""

# =============================================================================
# Imports
# =============================================================================
import os
import numpy as np
import pandas as pd
from statistics import mean, stdev
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from joblib import dump, load

from dna_featuregenerator import create_feature_vectors, create_feature_vectors2
from balanced_class import balanced_subsample

# =============================================================================
# Constants
# =============================================================================
SEQUENCES = [
    'Dark_Green',
    'Dark_Red',
    'Dark_Fred',
    'Dark_NIR',
    'Green_Red',
    'Green_Fred',
    'Green_NIR',
    'Red_Fred',
    'Red_NIR',
    'Fred_NIR'
]

COLOR_CLASSES = ['dark', 'green', 'red', 'fred', 'nir']

CLASSIFIER_FILENAME = 'classifiers_final.sav'
SVM_FILENAME = 'svms2.sav'
PREDICTION_OUTPUT_FILENAME = '16base_results.csv'
TOP_SEQUENCES_OUTPUT_FILENAME = 'top_1000_results.txt'
KEY_FILENAME = 'key.txt'
SEQUENCE_POOL_FILENAME = '../sequence generation/16base_sequences.txt'
FINAL_OUTPUT_FILENAME = 'selected_sequences.txt'
TRAINING_DATA_DIR = '../training data/'

NUM_CLASSIFIERS_PER_SEQUENCE = 10
TOP_K = 1000


# =============================================================================
# Step 1: Training and Evaluation
# =============================================================================
def train_and_evaluate(sequences):
    """
    Train SVM classifiers and evaluate with calibrated 10-fold CV.

    For each color pair, trains 10 classifiers on independent balanced
    subsamples. Each replicate is evaluated via 10-fold stratified CV
    using calibrated predictions (predict_proba >= 0.5). Final models
    are then trained on the full balanced sample for deployment.

    Returns:
        tuple: (list of SVMs, list of calibrated classifiers,
                list of per-comparison result dicts)
    """
    all_results = []
    classifiers = []
    svms = []

    print(f"{'Comparison':<16} {'Acc Mean':>10} {'Acc Std':>10} {'F1 Mean':>10} {'F1 Std':>10}")
    print("-" * 60)

    for pair in sequences:
        accuracy_list = []
        f1_list = []

        for rep in range(NUM_CLASSIFIERS_PER_SEQUENCE):
            features = create_feature_vectors(TRAINING_DATA_DIR + pair)
            X = features.drop('color', axis=1)
            y = features['color']
            xs, ys = balanced_subsample(X, y)

            skf = StratifiedKFold(n_splits=10, shuffle=False)
            fold_acc = []
            fold_f1 = []

            for train_idx, test_idx in skf.split(xs, ys):
                X_train, X_test = xs.iloc[train_idx], xs.iloc[test_idx]
                y_train, y_test = ys.iloc[train_idx], ys.iloc[test_idx]

                svm_fold = SVC(kernel='rbf', C=0.01, gamma='auto')
                svm_fold.fit(X_train, y_train)
                clf_fold = CalibratedClassifierCV(svm_fold, method='sigmoid', cv=5)
                clf_fold.fit(X_train, y_train)

                y_proba = clf_fold.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.5).astype(float)
                fold_acc.append(accuracy_score(y_test, y_pred))
                fold_f1.append(f1_score(y_test, y_pred))

            accuracy_list.append(mean(fold_acc))
            f1_list.append(mean(fold_f1))

            # Train final model on full balanced sample for deployment
            svm = SVC(kernel='rbf', C=0.01, gamma='auto')
            svm.fit(xs, ys)
            svms.append(svm)
            clf = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
            clf.fit(xs, ys)
            classifiers.append(clf)

        acc_mean = mean(accuracy_list)
        acc_std = stdev(accuracy_list)
        f1_mean = mean(f1_list)
        f1_std = stdev(f1_list)

        print(f"{pair:<16} {acc_mean:>10.4f} {acc_std:>10.4f} {f1_mean:>10.4f} {f1_std:>10.4f}")
        all_results.append({
            'Comparison': pair,
            'Accuracy_Mean': acc_mean,
            'Accuracy_StdDev': acc_std,
            'F1_Mean': f1_mean,
            'F1_StdDev': f1_std
        })

    return svms, classifiers, all_results


def save_classifiers(svms, classifiers):
    """Save trained SVMs and calibrated classifiers to disk."""
    dump(svms, SVM_FILENAME)
    dump(classifiers, CLASSIFIER_FILENAME)
    print(f"SVMs saved to {SVM_FILENAME}")
    print(f"Classifiers saved to {CLASSIFIER_FILENAME}")


# =============================================================================
# Step 2: Prediction
# =============================================================================
def predict_with_ensemble(classifiers, sequence_file, color_pairs):
    """
    Generate predictions using ensemble of calibrated classifiers.

    Probabilities are averaged across 10 classifiers for each color pair.
    All classifiers use the full 144 staple-motif features.

    Args:
        classifiers: List of trained calibrated classifiers
        sequence_file: Path to 16-base sequence file
        color_pairs: List of color pair names

    Returns:
        DataFrame with prediction probabilities for each color pair
    """
    print(f"Loading features from {sequence_file}...")
    seq_features = create_feature_vectors2(sequence_file)
    print(f"Loaded {len(seq_features)} sequences with {seq_features.shape[1]} features.")

    df = pd.DataFrame()
    classifier_idx = 0

    for idx, color_pair in enumerate(color_pairs):
        print(f"Processing {color_pair} ({idx + 1}/{len(color_pairs)})...")
        prob_class0 = None
        prob_class1 = None

        for j in range(NUM_CLASSIFIERS_PER_SEQUENCE):
            clf = classifiers[classifier_idx]
            proba = clf.predict_proba(seq_features)
            classifier_idx += 1

            if j == 0:
                prob_class0 = proba[:, 0]
                prob_class1 = proba[:, 1]
            else:
                prob_class0 = np.add(prob_class0, proba[:, 0])
                prob_class1 = np.add(prob_class1, proba[:, 1])

        prob_class0 = prob_class0 / NUM_CLASSIFIERS_PER_SEQUENCE
        prob_class1 = prob_class1 / NUM_CLASSIFIERS_PER_SEQUENCE

        df[color_pair] = prob_class0.tolist()
        df[color_pair + "_prob2"] = prob_class1.tolist()

    return df


def save_predictions(df, filename):
    """Save predictions to CSV file."""
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


# =============================================================================
# Step 3: Top-K Selection
# =============================================================================
COLOR_PAIR_TO_COLORS = {
    'Dark_Green': ('dark', 'green'),
    'Dark_Red': ('dark', 'red'),
    'Dark_Fred': ('dark', 'fred'),
    'Dark_NIR': ('dark', 'nir'),
    'Green_Red': ('green', 'red'),
    'Green_Fred': ('green', 'fred'),
    'Green_NIR': ('green', 'nir'),
    'Red_Fred': ('red', 'fred'),
    'Red_NIR': ('red', 'nir'),
    'Fred_NIR': ('fred', 'nir')
}


def compute_min_scores_per_class(df):
    """
    Compute minimum probability across color pairs for each color class.

    For each color class, collect all its probabilities from pairwise
    classifiers and take the minimum per sequence.

    Returns:
        dict: {color_class: list of min scores per sequence}
    """
    color_cols = {color: [] for color in COLOR_CLASSES}

    for pair, (color1, color2) in COLOR_PAIR_TO_COLORS.items():
        if pair in df.columns:
            color_cols[color1].append(pair)
        if pair + "_prob2" in df.columns:
            color_cols[color2].append(pair + "_prob2")

    min_scores = {}
    for color in COLOR_CLASSES:
        cols = color_cols[color]
        if cols:
            min_scores[color] = df[cols].min(axis=1).tolist()
            print(f"  {color}: {len(cols)} columns")
        else:
            print(f"Warning: No columns found for {color}")

    return min_scores


def select_top_k_per_class(min_scores, top_k=TOP_K):
    """
    Select top K sequences per color class.

    Each sequence is assigned to the class with the highest minimum
    probability. Then, for each class, the top K sequences (by score)
    are selected from those assigned to that class.

    Returns:
        dict: {color_class: {row_index: score}}
    """
    num_sequences = len(next(iter(min_scores.values())))
    top_selections = {color: {} for color in COLOR_CLASSES}

    for i in range(num_sequences):
        row_scores = {color: min_scores[color][i] for color in COLOR_CLASSES}
        dominant_class = max(row_scores, key=row_scores.get)
        score = row_scores[dominant_class]
        row_idx = i + 1

        if len(top_selections[dominant_class]) < top_k:
            top_selections[dominant_class][row_idx] = score
        elif score > min(top_selections[dominant_class].values()):
            worst_idx = min(top_selections[dominant_class], key=top_selections[dominant_class].get)
            del top_selections[dominant_class][worst_idx]
            top_selections[dominant_class][row_idx] = score

    for color in COLOR_CLASSES:
        print(f"  {color}: selected {len(top_selections[color])} sequences")

    return top_selections


def save_top_selections(top_selections, filename):
    """Save top selections to file."""
    with open(filename, 'w') as f:
        for color, selections in top_selections.items():
            f.write(f"=== {color.upper()} ===\n")
            for row_idx, score in sorted(selections.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{row_idx}\t{score}\n")
            f.write("\n")
    print(f"Top selections saved to {filename}")


def load_top_selections(filename):
    """Load top selections from file."""
    top_selections = {color: {} for color in COLOR_CLASSES}
    current_color = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("===") and line.endswith("==="):
                current_color = line.replace("=", "").strip().lower()
            elif line and current_color and '\t' in line:
                parts = line.split('\t')
                row_idx = int(parts[0])
                score = float(parts[1])
                top_selections[current_color][row_idx] = score

    print(f"Top selections loaded from {filename}")
    return top_selections


def save_key_indices(top_selections, filename):
    """Save all selected row indices to a key file."""
    all_indices = set()
    for selections in top_selections.values():
        all_indices.update(selections.keys())

    with open(filename, 'w') as f:
        for idx in sorted(all_indices):
            f.write(f"{idx}\n")
    print(f"Key indices saved to {filename}")


# =============================================================================
# Step 4: Sequence Identification
# =============================================================================
def identify_sequences_with_info(sequence_pool_filename, top_selections):
    """
    Find sequences in the pool file by their indices.

    Returns:
        list: List of tuples (sequence, color, score, row_index)
    """
    index_to_info = {}
    for color, selections in top_selections.items():
        for row_idx, score in selections.items():
            index_to_info[row_idx] = (color, score)

    selected_sequences = []
    with open(sequence_pool_filename, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if line_num in index_to_info:
                color, score = index_to_info[line_num]
                selected_sequences.append((line.strip(), color, score, line_num))

    return selected_sequences


def save_selected_sequences(sequences_with_info, filename):
    """Save selected sequences with color and probability."""
    sorted_sequences = sorted(sequences_with_info, key=lambda x: (x[1], -x[2]))

    with open(filename, 'w') as f:
        f.write("sequence\tcolor\tprobability\tindex\n")
        for seq, color, score, idx in sorted_sequences:
            seq_no_spaces = seq.replace(" ", "")
            f.write(f"{seq_no_spaces}\t{color}\t{score:.6f}\t{idx}\n")

    print(f"Selected sequences saved to {filename}")


# =============================================================================
# Pipeline Functions
# =============================================================================
def run_training_pipeline():
    """Step 1: Train classifiers with evaluation."""
    print("=== Step 1: Training Classifiers ===")
    svms, classifiers, results = train_and_evaluate(SEQUENCES)
    save_classifiers(svms, classifiers)

    results_df = pd.DataFrame(results)
    results_df.to_csv('training_cv_results.csv', index=False)
    print(f"CV results saved to training_cv_results.csv")

    overall_acc = mean([r['Accuracy_Mean'] for r in results])
    overall_f1 = mean([r['F1_Mean'] for r in results])
    print(f"\nOverall accuracy: {overall_acc:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Total classifiers: {len(classifiers)}")
    print("Training complete.\n")


def run_prediction_pipeline():
    """Step 2: Load pre-trained classifiers and predict."""
    print("=== Step 2: Generating Predictions ===")
    classifiers = load(CLASSIFIER_FILENAME)
    predictions_df = predict_with_ensemble(classifiers, SEQUENCE_POOL_FILENAME, SEQUENCES)
    save_predictions(predictions_df, PREDICTION_OUTPUT_FILENAME)
    print("Prediction complete.\n")
    return predictions_df


def run_selection_pipeline(results_filename):
    """Step 3: Select top K sequences per color class."""
    print("=== Step 3: Selecting Top 1000 per Class ===")
    df = pd.read_csv(results_filename)

    if 'remove' in df.columns:
        df = df.drop('remove', axis=1)

    min_scores = compute_min_scores_per_class(df)
    top_selections = select_top_k_per_class(min_scores)
    save_top_selections(top_selections, TOP_SEQUENCES_OUTPUT_FILENAME)
    save_key_indices(top_selections, KEY_FILENAME)
    print("Selection complete.\n")
    return top_selections


def run_identification_pipeline(top_selections=None):
    """Step 4: Identify sequences from the pool file."""
    print("=== Step 4: Identifying Sequences ===")

    if top_selections is None:
        if not os.path.exists(TOP_SEQUENCES_OUTPUT_FILENAME):
            print(f"ERROR: {TOP_SEQUENCES_OUTPUT_FILENAME} not found. Run Step 3 first.")
            return []
        top_selections = load_top_selections(TOP_SEQUENCES_OUTPUT_FILENAME)

    sequences_with_info = identify_sequences_with_info(SEQUENCE_POOL_FILENAME, top_selections)
    save_selected_sequences(sequences_with_info, FINAL_OUTPUT_FILENAME)
    print("Identification complete.\n")
    return sequences_with_info


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Uncomment the steps you want to run:

    # Step 1: Train new classifiers (with 10-fold CV evaluation)
    # run_training_pipeline()

    # Step 2: Load pre-trained classifiers and predict
    # run_prediction_pipeline()

    # Step 3: Select top 1000 sequences per class
    # run_selection_pipeline(PREDICTION_OUTPUT_FILENAME)

    # Step 4: Identify sequences
    # run_identification_pipeline()

    # Full pipeline (after training):
    # predictions_df = run_prediction_pipeline()
    # top_selections = run_selection_pipeline(PREDICTION_OUTPUT_FILENAME)
    # run_identification_pipeline(top_selections)

    print("Pipeline ready. Uncomment desired steps in __main__ to run.")
