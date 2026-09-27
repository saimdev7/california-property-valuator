# California Property Valuator

A FastAPI backend serving a Random Forest model trained on California housing census data, paired with a lightweight web interface for single-property and bulk CSV valuations.

## Features

- **Single estimate** — enter a block's income, housing stats, and coordinates to get an instant price estimate with a margin of error.
- **Bulk appraisal** — upload a CSV of multiple properties and download predictions for all of them at once.
- **REST API** — `/predict` and `/predict-file` endpoints for programmatic access.
- Model average error: **≈ $37,778**

## Tech Stack

- **Backend:** FastAPI, scikit-learn (Random Forest Regressor), pandas, joblib
- **Frontend:** Vanilla HTML/CSS/JS, served directly by FastAPI

## Project Structure

```
├── app/
│   └── main.py          # FastAPI app and API routes
├── models/
│   ├── price_predictor.joblib
│   └── house_feature.joblib
├── scripts/
│   ├── train.py          # Model training script
│   └── explore_dataset.py
├── static/
│   └── index.html         # Web app frontend
├── requirements.txt
```

## Setup

```bash
git clone https://github.com/saimdev7/california-property-valuator.git
cd california-property-valuator
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the App

```bash
cd app
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` for the web interface, or `http://127.0.0.1:8000/docs` for the interactive API docs.

## API Reference

**POST `/predict`** — single estimate

```json
{
  "MedInc": 4.5,
  "HouseAge": 25,
  "Averooms": 5.5,
  "Avebedrooms": 1.1,
  "Population": 1200,
  "AveOccupation": 3.0,
  "Latitude": 37.0,
  "Longitude": -120.0
}
```

**POST `/predict-file`** — bulk CSV upload

CSV must include columns: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`

## Model

Trained on California census block data using a Random Forest Regressor (see `scripts/train.py`). Model files are tracked with **Git LFS** due to size.

## Disclaimer

Estimates are model-generated and not a substitute for a formal real estate appraisal.
