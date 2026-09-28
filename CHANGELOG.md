# Changelog

## 2026-09-29 — Vercel and typography update

- Added Vercel Python function configuration, excluded training data, candidate artifacts, tests, and local-only files from the function bundle, and placed CDN-served assets under `public/`.
- Reduced `requirements.txt` to the pinned runtime packages; pandas remains because inference creates a DataFrame. The fresh environment loaded the final pipeline successfully.
- Switched to self-hosted Public Sans and Atkinson Hyperlegible WOFF2 fonts with their SIL OFL licenses, updated the requested headline and privacy line, and recorded the measured runtime/model bundle estimate.
- Replaced the README with the requested Vercel, preview, model, and limitations structure. Added desktop home and result screenshots supplied by the user; the existing form and validation-error captures remain. A mobile screenshot is still pending.

## 2026-09-28

- Added a local Direction A interface for the approved Set B inputs, with required-field and data-range validation, loading/empty/error states, and responsive layout.
- Kept the approved raw RandomForest pipeline and added Lower / Intermediate / Higher cutoffs (0.34 / 0.84) and copy to its JSON metadata. The prediction endpoint reads bands from that file.
- Added a CDC blood-pressure-equipment photograph with source and license credits; its fonts were later replaced by the 2026-09-29 typography update.
- Replaced `README.md` with the approved Set B documentation after resolving the earlier editor lock; retained the matching `README_SET_B.md` copy.
- Recomputed metrics from the repository's current grouped split. Earlier CV AUC values remain unreproducible because their fold assignments/predictions were not saved.
- Reported previously viewed holdout band counts as a secondary sanity check, not headline performance.

## 2026-09-28 — Public deployment preparation

- Replaced the repository-root `http.server` handler with Flask routes for `/`, `/health`, and `POST /api/predict`; static assets are now under `static/` and no other repository paths are served.
- Preserved required input and range checks, limited request bodies to 4 KiB, disabled Werkzeug access logs, and removed the raw prediction score from successful API responses.
- Added pinned Flask and Linux Gunicorn dependencies, recorded Python 3.12.14, and configured `PORT` plus local/deployed bind behavior.
- Added `.gitignore` rules for local environments, secrets, datasets, and candidate models while allowing only the final model and runtime JSON metadata.
- Added the UCI Heart Disease dataset citation and clinic-patient/screening limitations to both README copies; moved browser image and fonts under the Flask static directory.
