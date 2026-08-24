# Sensro

Sensro is a route prediction and data collection project for analyzing road conditions and generating route insights.

## Features

- Collect road-related data through the frontend interface
- Store and query data using SQLite
- Predict routes and route-related outputs using the ML pipeline
- Explore model training and comparison notebooks in the `ml/` folder
- Serve the project through the main application entry point

## Project Structure

- `main.py` – application entry point
- `data.py` – data handling logic
- `database.py` – database access and persistence
- `predict_route.py` – route prediction logic
- `ws_manager.py` – WebSocket-related management
- `frontend/` – web UI files
- `ml/` – notebooks, training scripts, and model artifacts
- `requirements.txt` – Python dependencies
- `requirements_ml.txt` – ML-specific dependencies

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

For ML-related features:

```bash
pip install -r requirements_ml.txt
```

## Run the app

```bash
python main.py
```

## Notes

This project includes a local SQLite database and a machine learning workflow under the `ml/` directory. You may need to ensure the required model files and data sources are present before running prediction or training tasks.

## License

This project is provided as-is for educational or personal use unless otherwise specified.
