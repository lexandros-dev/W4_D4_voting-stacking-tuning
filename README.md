# Voting, Stacking & Tuning

Bagging and boosting both build many models out of **one** algorithm. This repository takes
the other route: several models that already exist, combined into one prediction. The
combiner can be a fixed rule — a vote, an average — or it can be a second model **fitted**
to learn the combination, which is where stacking and blending live. The repository closes
on the search that finds good settings in the first place.

Two datasets carry it. **45,222 census records**, where the task is whether a person earned
more than $50,000, carries the classification half; its target is imbalanced, so F1 rather
than accuracy is the metric throughout. **20,640 California districts**, where the task is
the median house value, carries the regression half. Both ship in `data/`, so every notebook
runs on its own and offline.

```
01  voting and averaging: fixed rules, then a learned one   (census, classification)
        |
        v
02  blending and stacking: what the learned combiner costs  (california, regression)
        |
        v
03  optuna: searching the settings instead of guessing      (california, regression)
        |
        |  saves models/xgboost_tuned.json
        v
04  reading the model it found: permutation, PDP, SHAP      (no training at all)
```

## Learning Objectives

By the end of this repository, you should be able to:

- Combine several fitted classifiers with `VotingClassifier`, hard and soft, and state which
  of precision and recall a soft vote moves.
- Construct a stacked model from out-of-fold predictions, and a blended one from a holdout
  split, and decide which to use from the size of the table.
- Read a fitted meta-model's weights, including a negative one and one inflated only by a
  narrow column, and explain why a fixed averaging rule can use neither.
- Diagnose a failed ensemble from the members' error correlation and the spread of their
  scores.
- Tune five hyperparameters with an Optuna study, stop unpromising trials early from a
  training callback, and confirm the result on cross-validation before scoring the test set
  once.
- Interpret a fitted boosted model with permutation importance, partial dependence and SHAP,
  and state which question each one answers.

## Learning Path

**Scope** marks the core path every student is expected to complete. Optional lessons stay in
the repository and are worth returning to, but the day does not depend on them. **Units** are
indicative pacing, where one unit is 45 minutes.

| File / Folder | Description | Scope | Units |
| --- | --- | --- | --- |
| [**1 - Voting, Averaging and Stacking**](01_voting_stacking_classifiers.ipynb) | Five classifiers on the census data, two of them new. A majority vote fixes the 1,830 rows a minority got wrong and cannot touch the 1,374 a majority did. Soft voting beats the best member in all five folds; stacking wins on accuracy and loses on F1, because its meta-model minimizes a different loss. | Core | 2.25 |
| [**2 - Blending and Stacking Regressors**](02_stacking_lgbm_regressors.ipynb) | Five LightGBM settings whose errors correlate 0.910 to 0.969, so their average scores *worse* than the best of them, 0.4396 against 0.4383. A fitted combiner finds what the average cannot and gives the weakest member a **negative** weight. Blending holds back a quarter of the training rows to do it; stacking builds the same columns out-of-fold and pays nothing. | Core | 2.0 |
| [**3 - Hyperparameter Search with Optuna**](03_optuna_xgboost.ipynb) | Five XGBoost hyperparameters searched by a Bayesian sampler rather than guessed, comfortably beating the untouched defaults on the test districts. Trials report their score after every boosting round, so a sizeable share are stopped early; the winner is then confirmed on folds before the test set is touched once, and saved for notebook 4. | Core | 2.25 |
| [**4 - Making Sense of a Black Box**](04_interpreting_the_model.ipynb) | The saved model, questioned three ways and no training at all. XGBoost's free importance ranks a location dummy first; permutation importance on held-out districts ranks it sixth. Partial dependence gives direction and size, and SHAP attributes a single district's prediction to its columns exactly. The three rankings disagree, and each is right about its own question. | Core | 1.5 |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | `census.csv`: 45,222 records, 8 categorical and 5 numeric columns, with `income` as the target. `california_housing.csv`: 20,640 districts, 8 numeric columns and a 4-value `location`, with `MedHouseVal` as the target, in units of $100,000. Both are modified from their originals; see the references below. |
| [**Models**](models/) | `xgboost_tuned.json`, the model notebook 3's search settled on, saved so notebook 4 needs no training. Committed, so notebook 4 runs even if notebook 3 has not been. |
| [**Solutions**](solutions/) | Reference solutions for every notebook. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd amle-voting-stacking-tuning`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebook

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open `01_voting_stacking_classifiers.ipynb`, selecting the Python environment created by `uv sync` as the kernel. Work through the notebooks in order. All four are the core path.

> [!TIP]
> Notebooks 02, 03 and 04 pass `n_jobs=4` rather than `n_jobs=-1`. LightGBM's own documentation asks for **the number of real CPU cores, not the number of threads** — most CPUs run two threads per core — and warns against large thread counts on a small table, which 16,512 rows is. `-1` gives threads, so on a four-core laptop it asks for eight. Four is a conservative floor rather than a measurement: **count your own cores and raise it**, and remember that a laptop running a browser and a video call has fewer free than it has in total. Notebook 01 still uses `-1`, because there it runs five separate model fits side by side rather than threads inside one.

> [!CAUTION]
> macOS only: `uv sync` installs XGBoost and LightGBM, but neither bundles the system OpenMP runtime it links against. If a notebook reports that `libxgboost.dylib` or `lib_lightgbm.dylib` could not be loaded, install it once with `brew install libomp`. The Linux and Windows wheels ship OpenMP, so no extra step is needed there. See the [XGBoost](https://xgboost.readthedocs.io/en/stable/install.html) and [LightGBM](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) installation notes.

## References & Further Reading

- [**scikit-learn: Ensembles**](https://scikit-learn.org/stable/modules/ensemble.html): Voting and stacking alongside bagging and boosting, with the parameters that matter.
- [**scikit-learn: Naive Bayes**](https://scikit-learn.org/stable/modules/naive_bayes.html): The independence assumption notebook 01 introduces, and why it classifies well while estimating probabilities badly.
- [**scikit-learn: Nearest Neighbors**](https://scikit-learn.org/stable/modules/neighbors.html): Choosing `k`, the distance metrics available, and why the columns must be scaled first.
- [**Wolpert, Stacked Generalization, Neural Networks 5:241-259 (1992)**](https://doi.org/10.1016/S0893-6080(05)80023-1): The original, where training the combiner on out-of-sample predictions is the central point rather than a detail.
- [**Breiman, Stacked Regressions, Machine Learning 24:49-64 (1996)**](https://doi.org/10.1007/BF00117832): Stacking for regression, and the argument for constraining the meta-model's weights.
- [**Akiba et al., Optuna: A Next-generation Hyperparameter Optimization Framework (2019)**](https://doi.org/10.1145/3292500.3330701): The define-by-run API and the pruning strategy notebook 03 implements by hand.
- [**Optuna: Pruning**](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html): `trial.report()` and `trial.should_prune()`, and the pruners available.
- [**UCI: Adult**](https://archive.ics.uci.edu/dataset/2/adult): Extracted from the 1994 US census by Barry Becker and Ronny Kohavi, licensed CC BY 4.0. **Modified here:** the 48,842 original rows are reduced to the 45,222 with no missing values, two columns are renamed, and the leading spaces inside its category values have been stripped.
- [**California housing**](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset): Block groups from the 1990 US census, derived by Pace and Barry (1997) and distributed with scikit-learn as `fetch_california_housing`. **Modified here:** the eight numeric columns and the target are scikit-learn's, unchanged, with one column added. `location` is the `ocean_proximity` column of [Aurélien Géron's edition](https://github.com/ageron/handson-ml2/tree/master/datasets/housing), its values renamed to snake_case; the original's fifth value, `ISLAND`, does not appear in this file.
