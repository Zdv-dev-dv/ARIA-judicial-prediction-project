"""
grid_search_exp1.py
===================
Exhaustive hyperparameter search for the ECHR judicial prediction pipeline.

Runs sklearn GridSearchCV over TF-IDF and LinearSVC parameters for a single
article. The best parameters found by this script are hard-coded in
pipeline_exp1.ipynb for reproducibility.

Usage
-----
    python grid_search_exp1.py --article Article6 --path /path/to/crystal_ball_data/

    # Or edit ARTICLE and PATH in the __main__ block below.

Output
------
Results are written to time_results/<article>_time.txt (created automatically).

Dataset
-------
ECtHR Crystal Ball — Medvedeva, Vols & Wieling (2019)
https://github.com/masha-medvedeva/ECtHR_crystal_ball

Author
------
Zoé de Vries — ENS Paris-Saclay, ARIA programme (2023)
"""

from __future__ import print_function
import re, glob, sys, os, random, argparse
from time import gmtime, strftime, time
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
import warnings
from random import shuffle

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Pipeline and parameter grid
# ============================================================

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='word')),
    ('clf', LinearSVC())
])

parameters = {
    'tfidf__ngram_range': [(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)],
    'tfidf__lowercase':   (True, False),
    'tfidf__min_df':      (1, 2, 3),
    'tfidf__use_idf':     (False, True),
    'tfidf__binary':      (False, True),
    'tfidf__norm':        (None, 'l1', 'l2'),
    'tfidf__stop_words':  (None, 'english'),
    'clf__C':             (0.1, 1, 5)
}


# ============================================================
# Helper functions
# ============================================================

def balance(Xtrain, Ytrain):
    """
    Truncate the majority class so that violation and non-violation counts are equal.

    Parameters
    ----------
    Xtrain : list of str
    Ytrain : list of str

    Returns
    -------
    Xtrain, Ytrain : balanced lists
    """
    v  = [i for i, val in enumerate(Ytrain) if val == 'violation']
    nv = [i for i, val in enumerate(Ytrain) if val == 'non-violation']
    if len(nv) < len(v):
        v = v[:len(nv)]
    elif len(nv) > len(v):
        nv = nv[:len(v)]
    Xtrain = [Xtrain[j] for j in v] + [Xtrain[i] for i in nv]
    Ytrain = [Ytrain[j] for j in v] + [Ytrain[i] for i in nv]
    return Xtrain, Ytrain


def extract_text(starts, ends, cases, violation):
    """
    Extract text between two section-boundary patterns from a list of case files.

    Returns a list of (text, label, year) tuples.
    Cases without a parseable dateline (year == 0) are silently skipped.
    """
    facts = []
    D = []
    years = []
    for case in cases:
        contline = ''
        year = 0
        with open(case, 'r') as f:
            for line in f:
                dat = re.search(r'^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
                if dat is not None:
                    year = int(dat.group(2))
                    break
            if year > 0:
                years.append(year)
                wr = 0
                for line in f:
                    if wr == 0:
                        if re.search(starts, line) is not None:
                            wr = 1
                    if wr == 1 and re.search(ends, line) is None:
                        contline += line
                        contline += '\n'
                    elif re.search(ends, line) is not None:
                        break
                facts.append(contline)
    for i in range(len(facts)):
        D.append((facts[i], violation, years[i]))
    return D


def extract_parts(article, violation, part, path):
    """
    Extract a named structural section from case files matching a glob pattern.

    Parameters
    ----------
    article : str
        Article name (used only for logging).
    violation : str
        Label: 'violation' or 'non-violation'.
    part : str
        Section name. One of: facts, circumstances, relevant_law,
        procedure, procedure+facts.
    path : str
        Glob pattern for case files.

    Returns
    -------
    list of tuple : (text, label, year)
    """
    cases = glob.glob(path)
    facts = []
    D = []
    years = []

    if part == 'relevant_law':
        for case in cases:
            year = 0
            contline = ''
            with open(case, 'r') as f:
                for line in f:
                    dat = re.search(r'^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
                    if dat is not None:
                        year = int(dat.group(2))
                        break
                if year > 0:
                    years.append(year)
                    wr = 0
                    for line in f:
                        if wr == 0:
                            if re.search('RELEVANT', line) is not None:
                                wr = 1
                        if wr == 1 and re.search('THE LAW', line) is None and re.search('PROCEEDINGS', line) is None:
                            contline += line
                            contline += '\n'
                        elif re.search('THE LAW', line) is not None or re.search('PROCEEDINGS', line) is not None:
                            break
                    facts.append(contline)
        for i in range(len(facts)):
            D.append((facts[i], violation, years[i]))

    elif part == 'facts':
        D = extract_text('THE FACTS', 'THE LAW', cases, violation)
    elif part == 'circumstances':
        D = extract_text('CIRCUMSTANCES', 'RELEVANT', cases, violation)
    elif part == 'procedure':
        D = extract_text('PROCEDURE', 'THE FACTS', cases, violation)
    elif part == 'procedure+facts':
        D = extract_text('PROCEDURE', 'THE LAW', cases, violation)
    else:
        raise ValueError(f"Unknown part: '{part}'")

    return D


# ============================================================
# Main grid search pipeline
# ============================================================

def run_pipeline(article, part, path):
    """
    Load data, split by year, balance classes, run grid search, and evaluate.

    Training set: cases before 2014.
    Test set 1 (Xtest1): cases from 2014–2015.
    Test set 2 (Xtest2): cases from 2016 onwards.

    Parameters
    ----------
    article : str
    part : str
    path : str
        Root path to crystal_ball_data/.
    """
    v      = extract_parts(article, 'violation',     part, path + 'train/' + article + '/violation/*.txt')
    nv     = extract_parts(article, 'non-violation', part, path + 'train/' + article + '/non-violation/*.txt')
    test_v = extract_parts(article, 'violation',     part, path + 'test_violations/' + article + '/*.txt')

    # Balance the training portion
    if len(nv) < len(v):
        v = v[:len(nv)]
    elif len(nv) > len(v):
        nv = nv[:len(v)]

    trainset = v + nv + test_v
    Xtrain   = [i[0] for i in trainset]
    Ytrain   = [i[1] for i in trainset]
    YearTrain = [i[2] for i in trainset]

    # Split by year
    Xtest2, Ytest2 = [], []   # 2016+
    Xtest1, Ytest1 = [], []   # 2014–2015
    X, Y           = [], []   # pre-2014 (train)

    for i in range(len(YearTrain)):
        if YearTrain[i] >= 2016:
            Xtest2.append(Xtrain[i]); Ytest2.append(Ytrain[i])
        elif YearTrain[i] in (2014, 2015):
            Xtest1.append(Xtrain[i]); Ytest1.append(Ytrain[i])
        else:
            X.append(Xtrain[i]); Y.append(Ytrain[i])

    X, Y = balance(X, Y)
    Xtrain, Ytrain = X, Y
    Xtest1, Ytest1 = balance(Xtest1, Ytest1)
    Xtest2, Ytest2 = balance(Xtest2, Ytest2)

    X_1 = X + Xtest1
    Y_1 = Y + Ytest1

    # Build combined 2014–2017 test set (deduplicated via dict)
    # BUG FIX: original code used `for key, value in d_whole` — must be .items()
    d_whole = {}
    for i in range(len(Xtest1)):
        d_whole[Xtest1[i]] = Ytest1[i]   # also fixed: was Ytest1[1] (index bug)
    for i in range(len(Xtest2)):
        d_whole[Xtest2[i]] = Ytest2[i]

    X_3, Y_3 = [], []
    for key, value in d_whole.items():   # FIX: was `for key, value in d_whole`
        X_3.append(key); Y_3.append(value)
    X_3 = X_3[:len(Xtest1)]
    Y_3 = Y_3[:len(Xtest1)]

    print(f'Training on {len(Ytrain)} cases | '
          f'Test 2014-2015: {len(Ytest1)} | Test 2016+: {len(Ytest2)}')

    # Grid search
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    t0 = time()
    grid_search.fit(Xtrain, Ytrain)
    print(f'Grid search done in {time() - t0:.1f}s')
    print(f'Best CV score: {grid_search.best_score_:.3f}')
    print('Best parameters:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print(f'  {param_name}: {best_parameters[param_name]!r}')

    # Fit best model and evaluate
    vec = TfidfVectorizer()
    clf = LinearSVC()
    best_pipeline = Pipeline([('tfidf', vec), ('clf', clf)])
    best_pipeline.set_params(**best_parameters)
    best_pipeline.fit(Xtrain, Ytrain)

    print('\n--- Cross-validation on training set ---')
    Ypredict = cross_val_predict(best_pipeline, Xtrain, Ytrain, cv=3)
    _print_results(Ytrain, Ypredict)
    accuracies.append(accuracy_score(Ytrain, Ypredict))

    print('--- Test on 2014-2015 ---')
    _print_results(Ytest1, best_pipeline.predict(Xtest1))

    print('--- Test on 2016+ ---')
    _print_results(Ytest2, best_pipeline.predict(Xtest2))

    print('--- Test on 2014-2017 (combined) ---')
    _print_results(Y_3, best_pipeline.predict(X_3))

    print('--- Test on 2016+ (trained on train + 2014-2015) ---')
    best_pipeline.fit(X_1, Y_1)
    _print_results(Ytest2, best_pipeline.predict(Xtest2))

    print('--- Test on 2014-2015 (trained on train + 2016+) ---')
    X_2 = X + Xtest2; Y_2 = Y + Ytest2
    best_pipeline.fit(X_2, Y_2)
    _print_results(Ytest1, best_pipeline.predict(Xtest1))


def _print_results(Ytest, Ypredict):
    print('Accuracy:', accuracy_score(Ytest, Ypredict))
    print(classification_report(Ytest, Ypredict))
    print('Confusion matrix:\n', confusion_matrix(Ytest, Ypredict))
    print('_' * 40)


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Grid search for ECHR judicial prediction pipeline.'
    )
    parser.add_argument(
        '--article', type=str, default='Article6',
        help='Article to run grid search on (e.g. Article6). Default: Article6.'
    )
    parser.add_argument(
        '--path', type=str, default='../../crystal_ball_data/',
        help='Path to the crystal_ball_data/ directory.'
    )
    args = parser.parse_args()

    article = args.article
    path    = args.path

    parts = ['facts', 'circumstances', 'relevant_law', 'procedure', 'procedure+facts']
    accuracies = []

    os.makedirs('time_results', exist_ok=True)
    output_file = f'time_results/{article}_time.txt'

    print(f'Running grid search for {article}. Output -> {output_file}')

    with open(output_file, 'w') as f_out:
        sys.stdout = f_out
        current_time = strftime('%H:%M:%S', gmtime())
        print(f'Grid search started: {current_time}')
        for part in parts:
            print(f'\n{"="*50}\nTraining on section: {part}\n{"="*50}')
            run_pipeline(article, part, path)
        print('\nSections tested:', parts)
        print('Cross-val accuracies per section:', accuracies)

    sys.stdout = sys.__stdout__
    print(f'Done. Results written to {output_file}')
