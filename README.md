# SHL Grammar Scoring Assessment — Notebook Walkthrough

This README explains the full workflow implemented in the notebook `SHL_Grammar_Scoring_Assesment.ipynb` and documents each processing step from the basic baseline to the improvements that reduced RMSE. Follow these notes to reproduce the results or improve further.

---

## Overview

- Goal: Predict a continuous grammar quality score from audio responses.
- High-level approach: transcribe audio → extract linguistic features → train regression models → improve features (BERT embeddings + sentence-level features) → compare models → produce final submission.

## Files and artifacts

- Notebook: `SHL_Grammar_Scoring_Assesment.ipynb` (main workflow)
- Saved feature CSVs:
  - `train_features.csv`, `test_features.csv` (basic linguistic features)
  - `train_features_bert.csv`, `test_features_bert.csv` (added BERT embeddings)
  - `train_features_full.csv`, `test_features_full.csv` (final enhanced features)
- Submissions: `submission.csv`

## Environment & Setup

Install required packages (as used in the notebook):

```bash
pip install openai-whisper nltk scikit-learn librosa xgboost catboost lightgbm transformers
```

Notes:
- Whisper is used for speech-to-text (transcription). It may require significant CPU/GPU and disk resources.
- BERT embeddings require `transformers` and PyTorch (CPU/GPU) and are expensive to compute for many samples.

## Step-by-step walkthrough (progress from baseline → improved RMSE)

1) Data extraction and baseline features

  - The dataset archive is unzipped and CSV metadata files are loaded to enumerate `train` and `test` audio files.
  - Basic linguistic features are extracted per audio transcription (via Whisper):
    - `text` (raw transcription)
    - `total_words`, `num_nouns`, `num_verbs`, `num_adjs`, `avg_word_len`
  - These features form the initial feature matrix for modeling.

  Baseline modeling:

  - A `RandomForestRegressor` is trained on these basic features to produce a baseline MSE.
  - This step gives a first estimate of how well simple linguistic statistics predict the score.

2) Model expansion (boosting libraries)

  - Additional gradient-boosted regressors are trained on the same features:
    - `XGBRegressor` (XGBoost)
    - `CatBoostRegressor` (CatBoost)
    - `LGBMRegressor` (LightGBM)
  - Training MSEs are computed for each model to compare capacity and fit on the baseline features.

  Observation:

  - If a model shows near-zero training MSE (e.g., XGBoost/CatBoost in the notebook), that often indicates overfitting to the training set and not necessarily better generalization.

3) Feature augmentation with BERT embeddings

  - Each transcription `text` is tokenized and passed through `bert-base-uncased` to extract a 768-dim `[CLS]` embedding.
  - These embeddings are concatenated with the existing statistical features, producing `train_features_bert.csv` / `test_features_bert.csv`.
  - Models are re-trained using this enriched feature set (RandomForest, XGBoost, CatBoost, LightGBM).

  Effect on performance:

  - Adding dense semantic features (BERT) provides richer signals about fluency and content, often substantially improving predictive power when models can use the high-dimensional embedding effectively.
  - However, high-capacity models may overfit the enriched representation if not regularized or validated properly.

4) Additional linguistic feature engineering (sentence-level)

  - New hand-crafted features are computed from the `text` column:
    - `num_sentences` — sentence count (via `sent_tokenize`).
    - `avg_sentence_len` — average words per sentence.
    - `lexical_diversity` — type-token ratio (unique words / total words).
    - POS-based counts: `num_adverbs`, `num_pronouns`, `num_conjunctions`.
  - These features capture syntactic and cohesion cues not present in raw BERT embeddings or simple token counts.

5) Final training with enhanced features

  - The final feature set `train_features_full.csv` includes:
    - Original basic features
    - BERT embeddings (768 dims)
    - Sentence-level and POS-derived features
  - Models retrained on `X_train_bert` / `X_train_enhanced`
  - The notebook retrains RandomForest and LightGBM on the enhanced set and computes training MSEs.

6) Model selection and submission

  - The notebook compares all training MSEs and selects a best model. Due to risk of overfitting (zero MSE), the notebook prefers a robust choice (LightGBM with BERT features) even if XGBoost/CatBoost achieved very low training MSE.
  - Final predictions are generated for the test set and written to a CSV submission (e.g., `submission_final.csv`).

## Why each improvement helped RMSE (intuition)

- Baseline features (word counts, POS counts): capture surface-level fluency and grammar cues, giving a reasonable starting point.
- BERT embeddings: capture semantic content and contextual word usage — allows models to learn finer-grained differences correlated with grammar score.
- Sentence-level features: mark structural coherence and fluency (e.g., many short fragments suggest disfluency).
- Combining handcrafted features with deep embeddings gives complementary information: embeddings handle semantics; engineered features capture explicit grammatical and structural cues.

## Reproducibility (how to run)

1. Place the notebook and dataset in a working folder (same folder used in the notebook). The notebook expects the dataset in `/content/dataset` paths — update paths if running locally.

2. Install dependencies (repeat):

```bash
pip install openai-whisper nltk scikit-learn librosa xgboost catboost lightgbm transformers torch tqdm
python -m nltk.downloader punkt averaged_perceptron_tagger
```

3. Run cells in order. Important notes:

- Whisper transcription is expensive — consider running transcription once and saving `train_features.csv` / `test_features.csv` to avoid re-transcribing.
- BERT embeddings are expensive too; the notebook saves `train_features_bert.csv` to avoid recomputation.

4. To retrain models or try new ones, load the saved CSVs and re-run only the model-training sections.

## Practical tips to further reduce RMSE

- Cross-validation & early stopping: use K-fold CV or a validation set and early stopping (for boosting models) to prevent overfitting and get honest RMSE estimates.
- Hyperparameter tuning: use `RandomizedSearchCV` or `Optuna` to tune `n_estimators`, `max_depth`, `learning_rate`, etc.
- Dimensionality reduction: reduce BERT embedding size (PCA/UMAP) before training if model struggles with 768-dim features.
- Regularization: add L2 regularization or shrink learning rate for boosting models.
- Ensembling: average predictions from multiple well-validated models (LightGBM + RandomForest) to reduce variance.
- Data augmentation & cleansing: improve transcription quality (noise removal) and correct obvious transcription errors.

## Known caveats and notes

- Training MSE printed in the notebook is not the same as validation/test RMSE. Always validate on held-out data.
- Extremely low training MSE (0.0) indicates overfitting; do not trust such models without validation.
- Running the notebook end-to-end requires substantial compute (especially for Whisper and BERT). Consider using GPU-enabled runtime if available.

## Next steps (suggested experiments)

- Add a proper cross-validation loop and log validation RMSE for each model.
- Try fine-tuning a small transformer or a lightweight sentence embedding (e.g., `sentence-transformers`) instead of raw BERT CLS vectors.
- Experiment with additional linguistic features (e.g., error counts from grammar-checker tools, pause durations from audio metadata).
