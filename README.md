# Predicting Judicial Decisions of the European Court of Human Rights

---

## GENERAL INFORMATION

**Repository title:** Predicting Judicial Decisions of the European Court of Human Rights: A Machine Learning Pipeline

**Author:** Zoé de Vries  
**Contact:** zoe.de_vries@ens-paris-saclay.fr

**Affiliation:** ENS Paris-Saclay

**Associated report:**  
de Vries, Z. (2024). *Predicting Judicial Decisions of the European Court of Human Rights* [[ARIA research report]](https://github.com/Zdv-dev-dv/ARIA-judicial-prediction-project/blob/main/ARIA_RAPPORT_DE_VRIES.pdf). ENS Paris-Saclay.

**Licence:** MIT (code) — see `LICENSE`

---

## METHODOLOGICAL INFORMATION

### Research question

Can a Linear Support Vector Machine (LinearSVC) trained on TF-IDF representations of ECtHR case text reliably predict whether a given article of the European Convention on Human Rights has been violated? This pipeline replicates and extends the experimental design of Medvedeva et al. ([2019](https://link.springer.com/article/10.1007/s10506-019-09255-y), [2022](https://link.springer.com/article/10.1007/s10506-021-09306-3)) using a publicly available dataset of ECtHR judgements.

### Articles covered

Article 2, Article 3, Article 5, Article 6, Article 8, Article 10, Article 11, Article 13, Article 14.

### Pipeline overview

1. **Text extraction** — case documents are parsed by structural section (procedure, facts, circumstances, relevant law, or combinations thereof). Section boundaries are detected via regex against standard ECHR document headings.
2. **Vectorisation** — TF-IDF (word n-grams). Hyperparameters (ngram range, binary flag, IDF, normalisation, min_df, stop words, C) were tuned per article using exhaustive grid search (`grid_search_exp1.py`).
3. **Classification** — LinearSVC with article-specific optimised parameters.
4. **Evaluation** — 10-fold cross-validation; metrics reported: accuracy, precision, recall, F1-score (macro), confusion matrix.

### Hyperparameter tuning

`grid_search_exp1.py` runs `sklearn.model_selection.GridSearchCV` over the full parameter grid (see script header for full list) using a 3-fold CV on the training set. The best parameters are then applied in `pipeline_exp1.ipynb`.

### Software and environment

| Package | Version used |
|---------|-------------|
| Python  | 3.8+        |
| scikit-learn | ≥ 1.0 |
| nltk    | ≥ 3.7       |
| pandas  | ≥ 1.3       |

See `requirements.txt` for the full list. To install: `pip install -r requirements.txt`.

---

## DATA & FILE OVERVIEW

### Dataset

This project uses the **ECtHR Crystal Ball dataset** published by Medvedeva et al.:

> Medvedeva, M., Vols, M., & Wieling, M. (2019). Using machine learning to predict decisions of the European Court of Human Rights. *Artificial Intelligence and Law*, 28, 237–266.

**Dataset repository:** https://github.com/masha-medvedeva/ECtHR_crystal_ball  
**Licence:** see the dataset repository (publicly available for research use)

The dataset is **not included** in this repository. You must clone or download it separately. See [Data setup](#data-setup) below.

### File hierarchy

```
echr-judicial-prediction/
├── README.md                  # This file
├── LICENSE                    # MIT licence
├── requirements.txt           # Python dependencies
├── pipeline_exp1.ipynb        # Main classification pipeline (notebook)
└── grid_search_exp1.py        # Hyperparameter grid search script
```

### File descriptions

| File | Description |
|------|-------------|
| `pipeline_exp1.ipynb` | Jupyter notebook implementing the full classification pipeline. Loads data, extracts text by structural section, trains and evaluates a LinearSVC model with article-specific parameters determined by grid search. |
| `grid_search_exp1.py` | Standalone Python script for exhaustive hyperparameter search via `GridSearchCV`. Runs on a single article at a time; results inform the hard-coded parameters in the notebook. |

---

## DATA SETUP

1. Clone the dataset repository:
   ```bash
   git clone https://github.com/masha-medvedeva/ECtHR_crystal_ball.git
   ```

2. Note the path to the `crystal_ball_data/` folder inside that clone.

3. Open `pipeline_exp1.ipynb` and set the `DATA_PATH` variable in the first cell to that path:
   ```python
   DATA_PATH = "/path/to/ECtHR_crystal_ball/crystal_ball_data/"
   ```

4. For `grid_search_exp1.py`, set the `PATH` and `ARTICLE` variables in the configuration block at the top of the script.

---

## REPRODUCING THE RESULTS

### Full pipeline (notebook)

```bash
pip install -r requirements.txt
jupyter notebook pipeline_exp1.ipynb
```

Set `DATA_PATH` in the first cell, then run all cells. Results (accuracy, classification report, confusion matrix) are printed per article.

### Grid search (single article)

```bash
python grid_search_exp1.py --article Article6 --path /path/to/crystal_ball_data/
```

Or edit the `ARTICLE` and `PATH` variables directly in the `__main__` block. Output is written to `time_results/<article>_time.txt` (directory created automatically).

---

## DATA-SPECIFIC INFORMATION

### Input files (from ECtHR Crystal Ball dataset)

Each case is a plain-text `.txt` file structured according to standard ECHR document conventions. Files are organised as:

```
crystal_ball_data/
├── train/
│   └── ArticleN/
│       ├── violation/          # .txt files for violation cases
│       └── non-violation/      # .txt files for non-violation cases
└── test_violations/
    └── ArticleN/               # .txt files (violation cases only)
```

**Encoding:** UTF-8  
**Missing data:** Cases where no dateline could be parsed (year = 0) are silently skipped by the extraction functions. This affects a small minority of files.

### Output

Results are printed to stdout (notebook) or to `time_results/<article>_time.txt` (grid search script). No output files are committed to this repository as they are fully reproducible from the pipeline.

---

## ADDITIONAL INFORMATION

This code was written as part of the ARIA research programme at ENS Paris-Saclay (2024–2025), building on the experimental design of Medvedeva et al. (2019). The grid search script was used to determine the best TF-IDF and SVC parameters per article; those parameters are then fixed in the notebook for reproducibility and clarity.

A known limitation of this approach is that LinearSVC with TF-IDF features treats the classification task as a bag-of-words problem and does not capture semantic structure. Results should be interpreted with reference to the class imbalance present in the original dataset.

---

## REFERENCES

- Medvedeva, M., Vols, M., & Wieling, M. (2019). Using machine learning to predict decisions of the European Court of Human Rights. *Artificial Intelligence and Law*, 28, 237–266.
- Medvedeva, M., Wieling, M., & Vols, M. (2022). Rethinking the field of automatic prediction of court decisions. *Artificial Intelligence and Law*, 31, 195–212.
- Aletras, N., Tsarapatsanis, D., Preotiuc-Pietro, D., & Lampos, V. (2016). Predicting judicial decisions of the European Court of Human Rights: a natural language processing perspective. *PeerJ Computer Science*, 2, e93.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
