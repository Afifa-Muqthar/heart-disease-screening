# Heart Disease Screening Prototype

A local research prototype that compares five self-reported or home-measured inputs with records in the UCI heart-disease clinic dataset. It returns a dataset-similarity band. It does not diagnose heart disease or estimate population risk.

## Run locally

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Stop the server with Ctrl+C. Python **3.12.14** is recorded in `.python-version`. For a fresh environment, use your Python 3.12.14 interpreter to create `.venv` (`python -m venv .venv` if that interpreter is on `PATH`), then install dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### Public Linux deployment

Install the pinned dependencies with `python -m pip install -r requirements.txt`, then start the WSGI app with:

```sh
gunicorn --bind "0.0.0.0:${PORT:-8000}" --workers 2 --access-logfile /dev/null --error-logfile - app:app
```

The app uses the hosting platform's `PORT` value when set. Running `python app.py` locally binds to `127.0.0.1`; setting `APP_ENV=production` makes that entry point bind to `0.0.0.0`. Flask exposes only `/`, `/health`, `/api/predict`, and files under `/static/`. The request body is limited to 4 KiB.

The prediction endpoint accepts JSON at `POST /api/predict`. It returns only `band`, `band_name`, and `copy`; no numeric model score is returned. Every input is required. The app has no database or input logging, and the page states: “Answers are not saved.”

## Inputs

All five fields are required; the app does not fill missing user answers with defaults.

| Input | Allowed values |
|---|---|
| Age | Whole number 18–100. The model dataset covers 28–77; outside that range no score is produced. |
| Sex | Female / Male (categories in the dataset) |
| Chest-pain category (`cp`) | Typical angina / Atypical angina / Non-anginal / Asymptomatic |
| Exertion-related chest pain (`exang`) | Yes / No |
| Resting systolic blood pressure (`trestbps`) | 80–200 mmHg; a home cuff reading can be entered |

Example: age 54, Male, asymptomatic, no exertion-related discomfort, resting systolic pressure 130 mmHg. Example values do not imply a diagnosis. “Asymptomatic” is the source dataset's category label; it does not mean low risk or absence of disease.

## Model and evaluation

The approved Set B model is a `RandomForestClassifier` with 300 trees and `class_weight="balanced_subsample"`, in a scikit-learn preprocessing pipeline. The target is `num > 0`; `id` and `dataset` are excluded from model inputs. The raw 0–4 target is preserved. Missing values are imputed within training folds. The final serialized pipeline was refit on all 920 rows; band cutoffs and copy are stored beside it in `models/heart_disease_set_b_final.json`.

Grouped, stratified five-fold cross-validation used duplicate clinical predictor records as groups and `random_state=42`. At the approved threshold 0.3431, mean fold recall was **0.8575 ± 0.0271**, precision **0.7294 ± 0.0540**, and ROC-AUC **0.8086 ± 0.0598**. Quote **0.8086 ± 0.0598** as the current grouped-CV ROC-AUC: it is the mean and sample standard deviation of the five fold AUCs. Pooled training out-of-fold predictions give ROC-AUC 0.8041, recall 0.8575, and precision 0.7256. ROC-AUC does not depend on the decision threshold.

At threshold 0.3431, leave-one-source-site-out results (each model trained on the other sites) were:

| Held-out site | Recall | ROC-AUC |
|---|---:|---:|
| Cleveland (n=246) | 0.8496 | 0.7905 |
| Hungary (n=229) | 0.8675 | 0.8381 |
| Switzerland (n=96) | 0.8427 | 0.7568 |
| VA Long Beach (n=165) | 0.8770 | 0.6576 |

### Band cutoffs and limitations

The approved bands are Lower at score ≤0.34, Intermediate above 0.34 and below 0.84, and Higher at score ≥0.84. On current training out-of-fold score groups the observed rates were 58/254 (about 1 in 4), 149/235 (about 6 in 10), and 200/247 (about 8 in 10). These bands and rates were derived from out-of-fold predictions of models trained on only part of the data. The deployed model is refit on all 920 rows, so its scores may not have identical calibration or group rates. A band means similarity to this clinic-patient dataset, not personal probability, diagnosis, or population risk.

As a secondary sanity check, the 184-row holdout had been viewed during earlier candidate selection and is not an independent final evaluation. Using a training-partition-only Set B model with the rounded cutoffs grouped 68 records in Lower (13/68 observed disease), 65 in Intermediate (43/65), and 51 in Higher (46/51). These small selected clinic-record groups are descriptive only, not reliable population rates. The deployed estimator was subsequently refit on all 920 rows.

The five-fold split and holdout were both inspected during prior model selection, including candidate Set C and earlier feature sets. This creates selection bias; CV and site-held-out results are primary descriptive results here but are not a substitute for external validation. Site performance varies, the dataset is not population-representative, and calibration, clinical usefulness, subgroup fairness, and prospective safety have not been established. The model is not for clinical use.

ROC-AUC figures from earlier rounds do not reconcile with the current source and split. The earlier 0.808 was reported for a full-feature pipeline with omitted fields default-filled; 0.7952 ± 0.0389 was reported as a fold-mean for a five-input model, and 0.7942 as a pooled out-of-fold AUC. Re-running the current five-input pipeline with the fixed grouped training partition gives fold-mean 0.8086 ± 0.0598 and pooled 0.8041. The earlier values came from different folds or modeling details that were not preserved in the repository. Quote **0.8086 ± 0.0598** for the reproducible protocol documented here.

## Artifacts and assets

- Model: `models/heart_disease_set_b_final.joblib`
- Feature order, allowed values, threshold, bands, and band copy: `models/heart_disease_set_b_final.json`
- Publicly served browser files: `static/` only; image and font source/license details: `assets/CREDITS.md`
- `.gitignore` excludes `.venv`, environment files, local datasets, and all candidate model files. It allows only the final Set B model and its JSON metadata through the `models/` filter.

### Dataset credit and limitations

Dataset credit: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989), [Heart Disease dataset, UCI Machine Learning Repository](https://doi.org/10.24432/C52P4X), licensed CC BY 4.0. UCI documents four source databases: Cleveland, Hungary, Switzerland, and VA Long Beach. The repository's `dataset/heart_disease_uci.csv` is a local CSV copy; its exact extraction history was not verified.

This is a screening estimate from clinic-patient records, not a confirmed medical diagnosis or population-risk estimate. Dataset groups, validation limitations, and the fact that the displayed bands were defined from out-of-fold models while the deployed model was refit on all 920 rows are described above. Do not use the tool to make clinical decisions.

## Resume material

- Built a local Python/browser prototype around a five-input heart-disease Random Forest pipeline. **Interview backup:** Describe the five inputs and why the output is a dataset-comparison band rather than a diagnosis. **Rows:** [N].
- Evaluated the selected model with grouped stratified five-fold CV; recall was 0.8575 ± 0.0271 at threshold 0.3431. **Interview backup:** Explain why repeated clinical predictor records were grouped and why recall was prioritized. **Independent validation sites:** [N].
- Reported grouped-CV ROC-AUC of 0.8086 ± 0.0598 and leave-one-site-out metrics for four source sites. **Interview backup:** Explain fold-mean versus pooled out-of-fold AUC, and why the previously viewed holdout is secondary. **New external cohort size:** [N].
- Packaged the refit pipeline and band metadata as joblib and JSON artifacts. **Interview backup:** Explain the required-field validation and how approved cutoffs map scores to bands. **Prospective cases evaluated:** [N].

## Unresolved

- The model and user-facing bands have not been externally validated or clinically reviewed.
- The previously viewed holdout is not an unbiased final test set.
- Responsive and keyboard-focused UI behavior was implemented; no formal accessibility audit was run.
- No deployment, monitoring, or medical-device review has been performed.
