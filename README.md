# Comment Category Prediction

Classifying **198,000 user comments into 4 categories** (labels 0 to 3) using the comment
text together with engagement, identity-tag and time features. Built for a Kaggle
classification challenge as part of the IIT Madras BS Data Science program.

## Results (validation set, stratified 80/20 split)

| Model | Macro F1 | Accuracy |
|---|---|---|
| **LightGBM, tuned (GridSearchCV)** | **0.816** | **91.6%** |
| LightGBM, default | 0.799 | 91.1% |
| Logistic Regression (class balanced) | 0.793 | 89.8% |
| XGBoost | 0.778 | 89.8% |

Macro F1 is used because the classes are imbalanced: label 0 is the most common and label 3 the rarest,
so accuracy alone would hide poor performance on the small classes.

## Data

| | Rows | Columns |
|---|---|---|
| train.csv | 198,000 | 15 |
| test.csv | 102,000 | 14 |

Columns include the comment text, creation timestamp, post id, three emoticon counts, upvotes,
downvotes, two numeric indicator fields and identity tags (race, religion, gender, disability).
The identity tags are missing for about 73% of rows.

## Approach

**1. Exploratory analysis**
- Upvotes and downvotes are heavily right skewed; `log1p` makes them usable.
- Character length spikes near 1,000, a sign that long comments are truncated.
- Some comments are very long in characters but short in words, which points to spam-like text.
- Numeric features alone correlate weakly with the label (strongest: `if_2`, 0.23), so the text has to carry most of the signal.

**2. Feature construction**
- Engagement: total emoticons, log upvotes and downvotes, upvote ratio, log ratio
- Text shape: character length, word length, average word length
- Time: year, month, day of week and hour from the timestamp
- Missing identity tags filled as `unknown` instead of dropping rows

**3. Text processing**
- Lowercasing, whitespace normalisation, URL removal
- **TF-IDF** with 1-2 word n-grams, 40,000 features, English stop words removed, sublinear term frequency

**4. Model matrix**
- One-hot encoded categoricals + TF-IDF text + scaled numeric features, joined as one **sparse matrix**
  so 40,000+ columns fit comfortably in memory

**5. Modelling**
- Compared LightGBM, Logistic Regression and XGBoost on macro F1
- Tuned the top two with GridSearchCV; tuning improved LightGBM (0.799 to 0.816) but not Logistic Regression
- Final submission: a **LightGBM + Logistic Regression ensemble** (weighted blend of log probabilities)
  with per-class offsets that shift predictions toward the smaller classes

## Repository

```
comment_category_prediction.ipynb   full pipeline: EDA, features, models, tuning, submission
requirements.txt
```

## Run it

1. Add the competition data to a Kaggle notebook (or place `train.csv`, `test.csv` and `Sample.csv` locally and update the paths in the loading cell).
2. `pip install -r requirements.txt`
3. Run all cells. The notebook writes `submission.csv`.

Some model comparison and grid search cells are commented out because they take a long time;
their results are recorded in the markdown cells right below them.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SciPy (sparse matrices), Matplotlib, Seaborn
