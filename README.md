# Transfer Learning for 16-Base DNA-Stabilized Silver Nanoclusters

Machine learning pipeline using transfer learning to design 16-base DNA sequences for NIR-emitting DNA-stabilized silver nanoclusters.

## Overview

This project uses an SVM ensemble trained on curated 10-base DNA sequences to predict fluorescence properties of 16-base sequences. A transfer learning strategy adapts the model from the 10-base training domain to the 16-base prediction domain via feature-count normalization.

## Model Development

The core workflow uses an ensemble of **10x10 pairwise support vector machines (SVMs)** with an RBF kernel, calibrated with Platt scaling for probability output.

- **Kernel**: Radial basis function (RBF)
- **Parameters**: Regularization C = 0.01, gamma = auto (1/n_features = 1/144)
- **Features**: All 144 staple-motif features (no feature selection applied)
- **Calibration**: Platt scaling via `CalibratedClassifierCV(method='sigmoid', cv=5)`
- **Evaluation**: 10-fold stratified cross-validation using calibrated predictions (`predict_proba >= 0.5`)
- **Training**: 10 independent balanced subsamples per color pair, each producing one classifier

## Transfer Learning

Each DNA sequence is featurized using dinucleotide "staple features" -- ordered pairs at distances 1-9, yielding 144 features (16 pairs x 9 distances). To account for the change from 10 to 16 positions, each feature count is normalized by **(9-n)/(15-n)**, where n is the distance offset (0-8).

## Candidate Evaluation

Each candidate 16-base sequence is evaluated using the ensemble of calibrated classifiers trained on the curated 10-base dataset:

1. The probability of belonging to each color class (green, red, far-red, or NIR) is determined by aggregating probabilities across all pairwise classifiers that include that class
2. The **minimum probability** assigned to a class across its classifiers is taken as the confidence value for that class
3. Each sequence is assigned to the class with the highest minimum probability
4. Sequences assigned to the NIR class with the highest confidence are prioritized for synthesis

## Project Structure

```
transfer-learning-16base/
├── svm ensemble/
│   ├── DNA_SVM_predict_rbf.py    # Main pipeline script
│   ├── balanced_class.py         # Class balancing utilities
│   └── dna_featuregenerator.py   # Feature extraction for DNA sequences
├── training data/                 # 10-base training sequences by color pair
│   ├── Dark_Green
│   ├── Dark_Red
│   ├── Dark_Fred
│   ├── Dark_NIR
│   ├── Green_Red
│   ├── Green_Fred
│   ├── Green_NIR
│   ├── Red_Fred
│   ├── Red_NIR
│   └── Fred_NIR
└── sequence generation/
    └── 16base_generator          # Generate 16-base sequence pools
```

## Pipeline Steps

### Step 1: Training
Train 100 SVM classifiers (10 color pairs x 10 balanced subsamples). Each classifier is an RBF SVM calibrated with Platt scaling. Evaluation uses calibrated 10-fold CV per replicate.

### Step 2: Prediction
Load trained classifiers and generate predictions for 16-base sequences using ensemble averaging across 10 classifiers per color pair.

### Step 3: Selection
Select top 1000 sequences per color class (dark, green, red, far-red, NIR) based on minimum probability scores.

### Step 4: Identification
Look up selected sequences by their indices from the sequence pool file.

## Color Classes

- **dark** - No fluorescence
- **green** - Green fluorescence
- **red** - Red fluorescence
- **fred** - Far-red fluorescence
- **nir** - Near-infrared fluorescence

## Requirements

```
numpy
pandas
scikit-learn
joblib
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn joblib
```

## Usage

1. **Generate 16-base sequences** (optional):
   ```bash
   cd "sequence generation"
   python 16base_generator
   ```

2. **Run the pipeline**:
   ```bash
   cd "svm ensemble"
   python DNA_SVM_predict_rbf.py
   ```

3. **Customize**: Edit the `__main__` section in `DNA_SVM_predict_rbf.py` to uncomment desired steps.

## Output Files

- `classifiers_final.sav` - Trained calibrated classifiers
- `svms2.sav` - Trained SVM models
- `training_cv_results.csv` - Cross-validation accuracy and F1 per comparison
- `16base_results.csv` - Prediction probabilities for all sequences
- `top_1000_results.txt` - Top 1000 sequences per color class with scores
- `key.txt` - Row indices of selected sequences
- `selected_sequences.txt` - Final selected DNA sequences

## License

See LICENSE file for details.
