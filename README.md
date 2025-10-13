# ML_LABCA

---

## Quick start (run in Google Colab / local)

### Prerequisites
- Python 3.8+ (3.9 recommended)
- Google Earth Engine account (sign up at https://earthengine.google.com)
- If running locally: Node-less, but `earthengine-api` + `geemap` require authentication.

### Install dependencies
```bash
pip install -r requirements.txt
import ee
ee.Authenticate()
ee.Initialize()
python scripts/predict.py --aoi 12.9 72.6 13.1 72.8 --yield-csv data/Agricultural_Yield.csv
python scripts/predict.py

---

# 3) `methodology.md`

```markdown
# Methodology

## 1. Data sources
- Agricultural yield dataset: `data/Agricultural_Yield.csv`
- Crop recommendation dataset: `data/Crop_recommendation.csv`
- (Optional) pesticides dataset: `data/pesticides.csv`
- Satellite data: MODIS LST (MOD11A1) from Google Earth Engine

## 2. Preprocessing
- Trim and clean column names.
- Map crop names to numeric labels for classification.
- Fill missing numeric values with medians.
- Feature set used for classifier: N, P, K, temperature (AOI average LST), humidity, pH, rainfall.
- For yield lookup, `Yield` column used (units as in CSV).

## 3. Modeling
- Models: RandomForestClassifier (sklearn), XGBoost, VotingClassifier (hard voting).
- Scaling: MinMaxScaler for features.
- Split: train_test_split with stratify on crops.
- Evaluation: accuracy_score and 5-fold cross-validation.

## 4. Prediction & Yield Lookup
- For a user-specified AOI, fetch mean LST for last 7 days from MODIS via Earth Engine.
- Accept user agronomic inputs (N,P,K, humidity, pH, rainfall).
- Predict recommended crop using the trained voting classifier.
- Lookup historical yield info for the predicted crop from `Agricultural_Yield.csv` and print average & best state.

## 5. Notes on Forecast and Dashboard
- The dataset and predicted counts can be exported to CSV for Power BI consumption.
- Power BI forecasting requires a continuous X-axis (e.g., temperature or year).
