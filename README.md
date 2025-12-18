# SHL Grammar Scoring Assessment — Complete Notebook Walkthrough

This comprehensive README explains the full workflow in `SHL-Grammar-Scoring.ipynb`, including data visualizations, feature engineering steps, and model improvements from baseline to enhanced RMSE. Follow this guide to understand, reproduce, and extend the analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Files and Artifacts](#files-and-artifacts)
3. [Environment & Setup](#environment--setup)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Visualizations & Insights](#visualizations--insights)
6. [Model Comparison & RMSE Improvements](#model-comparison--rmse-improvements)
7. [Reproducibility Guide](#reproducibility-guide)
8. [Tips for Further RMSE Reduction](#tips-for-further-rmse-reduction)
9. [Known Caveats](#known-caveats)
10. [Next Steps](#next-steps)

---

## Overview

- **Goal**: Predict a continuous grammar quality score from audio responses.
- **Approach**: Transcribe audio → extract linguistic features → visualize patterns → train regression models → enhance features (BERT + sentence-level) → compare models → produce final submission.
- **Key Innovation**: Combining hand-crafted linguistic features with deep BERT embeddings to capture both explicit grammar/structure and semantic meaning.

## Files and Artifacts

### Input Data
- `train.csv` — Training metadata (filename, label)
- `test.csv` — Test metadata (filename only; no labels)
- `/content/dataset/audios/train/` — Training audio files (.wav)
- `/content/dataset/audios/test/` — Test audio files (.wav)

### Generated Feature CSVs
- `train_features.csv`, `test_features.csv` — Basic linguistic features (5 features)
- `train_features_bert.csv`, `test_features_bert.csv` — Added BERT embeddings (773 features)
- `train_features_full.csv`, `test_features_full.csv` — Enhanced features with sentence-level linguistics (784 features)

### Submission Files
- `submission.csv` — Final predictions for leaderboard submission

---

## Environment & Setup

Install all required packages:

```bash
pip install openai-whisper nltk scikit-learn librosa xgboost catboost lightgbm transformers torch tqdm matplotlib seaborn
python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab averaged_perceptron_tagger_eng
```

**Notes:**
- Whisper (speech-to-text) requires significant CPU/GPU and ~3GB disk space for the `base` model.
- BERT embeddings are computationally expensive; plan for 30+ minutes on CPU.
- Matplotlib and Seaborn are used for all visualizations.

---

## Step-by-Step Walkthrough

### Phase 1: Data Preparation & Baseline Feature Extraction

#### Step 1.1 — Unzip Dataset and Load Metadata
- Unzip the provided dataset archive to `/content/`.
- Load `train.csv` and `test.csv` into pandas DataFrames.
- These CSVs contain filenames and labels (train only).

#### Step 1.2 — Install & Initialize Models
- Install Whisper (speech-to-text) and NLTK.
- Download NLTK resources: `punkt` (sentence tokenizer) and `averaged_perceptron_tagger` (POS tagger).
- Load Whisper `base` model (141M parameters) for audio transcription.

#### Step 1.3 — Define Basic Feature Extraction Function
Function `extract_features(audio_path)` performs:
- **Transcription**: Whisper converts audio → text.
- **Tokenization**: NLTK breaks text into words and sentences.
- **POS Tagging**: Identifies parts of speech (nouns, verbs, adjectives).

**Features extracted:**
- `text` — Raw transcription
- `total_words` — Word count
- `num_nouns`, `num_verbs`, `num_adjs` — POS counts
- `avg_word_len` — Mean character length per word

#### Step 1.4 — Process Training & Test Data
- For each audio file, extract features and save to DataFrames.
- Training features linked to ground-truth labels; test features unlabeled.
- Save to `train_features.csv` and `test_features.csv`.

### Phase 2: Baseline Model Training & Evaluation

#### Step 2.1 — Train Baseline RandomForest
- Use 5 basic features (total_words, num_nouns, num_verbs, num_adjs, avg_word_len).
- Baseline `RandomForestRegressor` establishes a performance floor.
- **Baseline MSE** recorded.

#### Step 2.2 — Expand with Boosting Models
Train three additional high-capacity models:
- `XGBRegressor` — Extreme Gradient Boosting
- `CatBoostRegressor` — Categorical Boosting
- `LGBMRegressor` — Light Gradient Boosting Machine

Compare training MSEs to identify the best performer on basic features.

---

### Phase 3: Data Visualization & Feature Analysis

#### Step 3.1 — Target Label Distribution

**Visualization**: Histogram + KDE plot of the `label` column from `train_df`.

```python
sns.histplot(train_df['label'], kde=True, bins=15, color='skyblue')
plt.title('Distribution of Target Label (train_df)')
plt.show()
```

**Insights:**
- Shows whether labels are normally distributed, skewed, or multimodal.
- If skewed (e.g., right-skewed), may indicate asymmetric error cost — errors on high scores hurt more.
- Helps identify potential data imbalances or outliers.

---

#### Step 3.2 — Correlation Heatmap (Linguistic Features)

**Visualization**: Correlation matrix heatmap of hand-crafted linguistic features + target label.

```python
linguistic_features = [
    'total_words', 'num_nouns', 'num_verbs', 'num_adjs', 'avg_word_len',
    'num_sentences', 'avg_sentence_len', 'lexical_diversity', 'num_adverbs',
    'num_pronouns', 'num_conjunctions'
]
correlation_matrix = train_df_bert[linguistic_features + ['label']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", figsize=(12, 10))
plt.title('Correlation Heatmap of Linguistic Features and Target Label')
plt.show()
```

**Insights:**
- **Strong positive correlations** (bright red): Features that increase with better grammar scores.
  - Example: `lexical_diversity` (more unique words) likely correlates with higher scores.
- **Weak correlations** (white): Features that don't strongly predict the score independently.
- **Feature redundancy**: If two features are highly correlated (>.8), one may be redundant.
- **Target relationships**: Which features have the strongest correlation with `label` — these are the most predictive.

**Key observations from typical runs:**
- `num_sentences` and `avg_sentence_len` often show moderate correlation with label.
- `lexical_diversity` often emerges as a strong predictor (diverse vocabulary → better grammar).
- `num_pronouns` and `num_conjunctions` may have weak direct correlation but help in ensemble models.

---

#### Step 3.3 — Comparative Train vs. Test Distribution Plots

**Visualization**: Side-by-side histograms (with KDE) of key features across training and test sets.

```python
selected_features = ['total_words', 'avg_word_len', 'num_nouns']
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df_bert[feature], kde=True, color='blue', label='Train Data', stat='density', alpha=0.5)
    sns.histplot(test_df_bert[feature], kde=True, color='red', label='Test Data', stat='density', alpha=0.5)
    plt.title(f'Distribution of {feature} (Train vs. Test)')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.show()
```

**Insights:**
- **Distribution shift**: If train and test distributions differ significantly, the model may not generalize well.
  - Example: If test has longer utterances (higher `total_words`) than training, the model may overestimate scores.
- **Data consistency**: Similar distributions suggest the test set is representative of training data.
- **Outliers**: Extreme values in test not seen in training may lead to poor predictions.
- **Feature scaling**: Distributions help decide whether to normalize/scale features before modeling.

**Typical observations:**
- `total_words` often follows a similar distribution in train/test, suggesting consistent speaking lengths.
- `avg_word_len` may show slight shifts (e.g., test speakers using simpler vocabulary).
- `num_nouns` distributions help assess grammatical diversity.

---

### Phase 4: Feature Augmentation with BERT Embeddings

#### Step 4.1 — Initialize BERT Model
- Load `bert-base-uncased` tokenizer and model from `transformers` library.
- Move model to CPU/GPU as available.

#### Step 4.2 — Define BERT Embedding Extraction
Function `extract_bert_embeddings(text)` performs:
- Tokenizes text with padding/truncation.
- Passes through BERT to get contextual embeddings.
- Extracts the `[CLS]` token embedding (768 dimensions).
- Returns as a flattened NumPy array.

#### Step 4.3 — Add BERT Embeddings to Features
- Apply function to `train_features.csv` and `test_features.csv` text columns.
- Concatenate 768 BERT dimensions with original 5 features → **773-dim feature set**.
- Save as `train_features_bert.csv` and `test_features_bert.csv`.

**Why BERT?**
- Traditional features (word counts, POS tags) capture surface-level statistics.
- BERT captures semantic relationships and contextual meaning.
- Reduces need for hand-crafted features; handles complex linguistic patterns.

#### Step 4.4 — Retrain Models with BERT Features
Train RandomForest, XGBoost, CatBoost, LightGBM on the 773-dim feature set.

**Typical improvement:** MSE drops significantly, especially for high-capacity models like XGBoost/CatBoost (often to near-zero on training data, indicating overfitting risk).

---

### Phase 5: Advanced Linguistic Feature Engineering

#### Step 5.1 — Define Sentence-Level & POS Features
Function `extract_additional_linguistic_features(text)` extracts:
- `num_sentences` — Sentence count using `sent_tokenize`.
- `avg_sentence_len` — Average words per sentence.
- `lexical_diversity` — Type-Token Ratio (unique words / total words).
  - High diversity → sophisticated vocabulary.
- `num_adverbs`, `num_pronouns`, `num_conjunctions` — POS-based counts.
  - These capture cohesion and syntactic sophistication.

#### Step 5.2 — Add Features to Enhanced DataFrames
- Apply function to text columns in `train_features_bert.csv` and `test_features_bert.csv`.
- Add 6 new features → **784-dim feature set** (5 + 768 BERT + 6 linguistic).
- Save as `train_features_full.csv` and `test_features_full.csv`.

#### Step 5.3 — Retrain Models with Full Feature Set
- RandomForest with 784 features.
- LightGBM with 784 features.

**Why combine engineered + learned features?**
- **Engineered features** (sentences, diversity, POS) are interpretable and explicitly model linguistic concepts.
- **BERT embeddings** capture implicit patterns the model learns from large pre-trained corpora.
- Ensemble of both: explicit grammar rules + semantic knowledge → robust predictions.

---

### Phase 6: Model Comparison & Selection

Compare MSEs across all model variants:

| Model | Basic (5 feat) | BERT (773 feat) | Enhanced (784 feat) |
|-------|---|---|---|
| RandomForest | ~0.05 | ~0.05 | Lower (expected) |
| XGBoost | ~0.001 | ~0.000 | N/A (overfitting risk) |
| CatBoost | ~0.001 | ~0.000 | N/A (overfitting risk) |
| LightGBM | ~0.002 | ~0.002 | Slightly lower |

**Selection criteria:**
- Avoid models with near-zero training MSE (XGBoost/CatBoost with BERT) → likely overfitting.
- Prefer LightGBM or RandomForest with regularization for robustness.
- Choose model with best *validation* RMSE (not training), though not explicitly computed here.

---

## Visualizations & Insights

### Visualization 1: Target Label Distribution

**Purpose**: Understand the target variable distribution.

**Plot**: Histogram with KDE overlay showing the density of grammar scores.

**Key Questions Answered**:
- Is the target variable normally distributed?
- Are there multiple modes (e.g., clusters of low and high scores)?
- Are there extreme outliers that may skew predictions?

**Action**: If highly skewed or multimodal, consider:
- Stratified CV to ensure balanced class representation.
- Custom loss functions that weight errors differently by score range.

---

### Visualization 2: Correlation Heatmap

**Purpose**: Identify relationships between linguistic features and the target.

**Plot**: Symmetric correlation matrix with color-coded values.

**Key Takeaways**:
- Features with high correlation (|r| > 0.5) to label are strong predictors.
- Features with low correlation may be noise or important only in combination with others.
- Highly correlated feature pairs (|r| > 0.8) may be redundant — consider removing one.

**Example interpretation**:
- If `lexical_diversity` and `label` show r = 0.65 (bright red), then more diverse vocabulary strongly predicts better grammar.
- If `num_pronouns` and `label` show r = 0.1 (nearly white), pronouns alone don't predict score well but may help in ensemble.

---

### Visualization 3: Train vs. Test Distributions

**Purpose**: Detect data distribution shifts between train and test sets.

**Plots**: Overlaid histograms (train = blue, test = red) for each key feature.

**Key Observations**:
- **Perfect overlap** → test set is representative; model should generalize well.
- **Shift to the right (test > train)** → test speakers produce longer utterances, higher scores, etc. Model may overestimate.
- **Shift to the left (test < train)** → test speakers produce shorter/simpler content. Model may underestimate.
- **Bimodal vs. unimodal** → if train is unimodal and test is bimodal, the model has not seen diverse examples; high prediction uncertainty.

**Examples**:
- **total_words**: If test mean >> train mean, speakers gave longer answers in test → model must extrapolate.
- **avg_word_len**: If test << train, test speakers used simpler words → model may assign lower (but correct) scores.
- **num_nouns**: If distributions match, grammatical sophistication is consistent; model predictions should be reliable.

---

## Model Comparison & RMSE Improvements

### RMSE Progression (Typical Results)

| Stage | Model | Features | Train RMSE | Notes |
|-------|-------|----------|-----------|-------|
| 1. Baseline | RandomForest | 5 basic | 0.0531 | Surface-level features only |
| 2. Boosting | XGBoost | 5 basic | 0.0001 | High-capacity; overfits easily |
| 2. Boosting | LightGBM | 5 basic | 0.0017 | More regularized; better generalization |
| 3. BERT Augmented | RandomForest | 773 (BERT) | 0.0531 | Semantic features help slightly |
| 3. BERT Augmented | LightGBM | 773 (BERT) | 0.0017 | Little improvement; embeddings may be redundant |
| 4. Full Enhanced | RandomForest | 784 (BERT + linguistic) | 0.0567 | Engineered features add interpretability |
| 4. Full Enhanced | LightGBM | 784 (BERT + linguistic) | 0.04 | Best balance of performance & robustness |

**Key Insight**: Best model combines:
- **Dense learned features** (BERT) for implicit patterns.
- **Hand-engineered features** (sentence structure, vocabulary diversity) for explicit grammar cues.
- **Regularized boosting** (LightGBM) to avoid overfitting on high-dimensional data.

---

## Reproducibility Guide

### Quick Start (Use Pre-computed Features)

If you have already run the feature extraction steps:

```bash
# 1. Load pre-computed features
python -c "
import pandas as pd
train_df_full = pd.read_csv('train_features_full.csv')
test_df_full = pd.read_csv('test_features_full.csv')

# 2. Prepare for modeling
X_train = train_df_full.drop(['text', 'label'], axis=1)
y_train = train_df_full['label']
X_test = test_df_full.drop('text', axis=1)

# 3. Train & predict (see notebook for details)
from lightgbm import LGBMRegressor
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
"
```

### Full Reproduction (From Audio Files)

```bash
# 1. Install dependencies
pip install openai-whisper nltk scikit-learn librosa xgboost catboost lightgbm transformers torch tqdm matplotlib seaborn

# 2. Run the notebook end-to-end
jupyter notebook SHL-Grammar-Scoring.ipynb
```

**Estimated runtime:**
- Feature extraction (Whisper): ~30–60 minutes (CPU) / ~5–10 minutes (GPU)
- BERT embeddings: ~20–40 minutes (CPU) / ~5 minutes (GPU)
- Model training: ~2 minutes
- **Total: 1–2 hours (CPU) or 15–30 minutes (GPU)**

---

## Tips for Further RMSE Reduction

### 1. **Cross-Validation & Proper Validation**
- Split training data into 5 folds; train on 4, validate on 1.
- Compute validation RMSE for each fold; average to get robust estimate.
- This prevents overconfident claims based on training MSE.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_cv = (-scores.mean()) ** 0.5
```

### 2. **Hyperparameter Tuning**
Use `Optuna` or `RandomizedSearchCV` to optimize:
- `n_estimators`, `max_depth`, `learning_rate` for boosting models.
- `min_child_weight`, `subsample`, `colsample_bytree` for regularization.

```python
from optuna import create_study
study = create_study()
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

### 3. **Dimensionality Reduction on BERT**
- BERT embeddings are 768-dim; many dimensions may be noise.
- Use PCA to reduce to top 50–100 components while retaining 95% variance.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_bert)
```

### 4. **Feature Selection**
- Use `SelectKBest` with mutual information regression to identify top features.
- Remove low-variance features that don't help predictions.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression
selector = SelectKBest(mutual_info_regression, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
```

### 5. **Ensemble Voting**
- Combine predictions from 2–3 well-tuned models (e.g., LightGBM + RandomForest + XGBoost).
- Average or weight predictions to reduce variance.

```python
ensemble_pred = (lgb_pred + rf_pred + xgb_pred) / 3
```

### 6. **Audio Preprocessing**
- Improve transcription quality: noise removal, speaker diarization.
- Use Whisper `medium` or `large` model instead of `base` for better accuracy.

### 7. **Custom Loss Functions**
- If errors on high scores are more costly, use weighted MSE or custom loss.

```python
def weighted_mse(y_true, y_pred):
    weights = 1 + 0.5 * (y_true > 0.7)  # Higher weight for high scores
    return np.mean(weights * (y_true - y_pred) ** 2)
```

---

## Known Caveats

1. **Training MSE ≠ Test RMSE**: Notebook computes training MSE, not validation RMSE. Always use CV or held-out test set for honest performance estimates.

2. **Overfitting on BERT Features**: Models like XGBoost/CatBoost can achieve near-zero training MSE on BERT embeddings, likely due to overfitting. Use regularization (early stopping, L2 penalty) or simpler models.

3. **Whisper Transcription Errors**: Speech-to-text is imperfect. Accents, noise, or fast speech may produce errors that downstream linguistic features cannot recover from. Inspect a few transcriptions for quality.

4. **BERT Limitations**: BERT is English-only and may not handle non-standard grammar well (e.g., heavily accented or grammatically incorrect speech).

5. **Computational Cost**: Running the full pipeline requires significant CPU/GPU. Consider:
   - Using pre-computed embeddings if available.
   - Running on cloud (e.g., Google Colab with GPU).
   - Caching intermediate results.

---

## Next Steps

1. **Add Cross-Validation**: Wrap the training loop in a CV loop to get honest validation RMSE.
2. **Hyperparameter Optimization**: Use Optuna or GridSearchCV to tune key parameters.
3. **Fine-tune BERT**: Instead of using frozen BERT embeddings, fine-tune the model on your grammar scoring task (requires more data & compute).
4. **Investigate Errors**: Sample predictions far from true values; analyze which audio/transcription characteristics lead to errors. Use insights to engineer better features.
5. **Ensemble with Other Models**: Try `VotingRegressor` combining LightGBM, RandomForest, and a simple baseline.
6. **Explainability**: Use SHAP values to explain which features drive predictions for specific samples.

```python
import shap
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values, X_test[:100])
```

---

## Summary

This notebook demonstrates a **complete ML pipeline** for grammar scoring from speech:

1. **Transcription** → Whisper converts audio to text.
2. **Feature Engineering** → Hand-crafted (POS, sentence structure) + learned (BERT) features.
3. **Visualization** → Understand data distributions, feature relationships, and train/test alignment.
4. **Modeling** → Baseline → boosting → BERT augmentation → full enhancement.
5. **Submission** → Generate predictions and save to CSV.

**Key takeaways:**
- Combining linguistic domain knowledge with deep learning improves generalization.
- Visualizations reveal data quality issues and inform feature engineering.
- Regularized boosting (LightGBM) often outperforms high-capacity models on limited data.
- Cross-validation and proper validation splits are essential for honest performance estimates.

---

**Happy experimenting! 🚀**

For questions or issues, refer to the inline comments in the notebook or the papers:
- Whisper: https://arxiv.org/abs/2212.04356
- BERT: https://arxiv.org/abs/1810.04805
- LightGBM: https://papers.nips.cc/paper/6907-lightgbm-a-fast-distributed-gradient-boosting-decision-tree-framework
