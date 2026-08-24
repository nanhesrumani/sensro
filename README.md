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
- `data.py` – raw data handling and preprocessing
- `database.py` – SQLite database access and persistence
- `predict_route.py` – prediction logic for routes
- `ws_manager.py` – WebSocket manager for live communication
- `frontend/` – browser UI assets and pages
- `generate_notebooks.py` – notebook-generation helper
- `manifest.json` – project metadata/configuration
- `road_data.db` – local SQLite database file
- `road_data.sqbpro` – SQLite database project file
- `ml/` – exploration, training, prediction scripts, and model outputs
- `requirements.txt` – base Python dependencies
- `requirements_ml.txt` – ML-specific dependencies
- `README.md` – project overview

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

## Hosting the frontend (GitHub Pages)

- The static frontend in `frontend/` is published to GitHub Pages (gh-pages branch).
- GitHub Pages serves only static files — it cannot run the Python backend.
- The frontend is implemented to prefer the live backend API (`/api/...`) and falls back to a snapshot file `/segments.json` when the backend is unavailable.

To publish the frontend (push the `frontend/` folder to the `gh-pages` branch):

```bash
git subtree push --prefix frontend origin gh-pages
```

## Snapshot workflow (read-only Pages site + on-demand backend)

1. Run your backend (locally or on a host) when you want to accept writes:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. Export the current segments (either from the running API or directly from the local DB):

- Using the running backend API:

```bash
python3 tools/export_segments.py --backend http://localhost:8000
```

- Or export directly from the local SQLite DB (`road_data.db`):

```bash
python3 tools/export_segments_from_db.py --db road_data.db --out frontend/segments.json
```

3. Commit and publish the snapshot so GitHub Pages shows the latest data:

```bash
git add frontend/segments.json
git commit -m "Snapshot: update segments.json"
git subtree push --prefix frontend origin gh-pages
```

Notes:
- While the backend is offline the site is read-only and uses `segments.json` for historical data.
- To accept live writes or WebSocket alerts, the backend must be running and reachable (CORS for `https://<your-username>.github.io` is already added in `main.py`).

## Deploying the backend (recommended options)

- Use Render, Vercel, Fly, Railway, or a small VM to host the Python FastAPI service. Example start command for hosts that provide `$PORT`:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

- If you deploy, update the frontend to point to the deployed backend URL (or keep using relative `/api` if you proxy or host backend under the same domain). You can also keep the snapshot workflow running for Pages.

### Quick deploy to Fly (recommended alternative)

This repository includes a `Dockerfile` and a GitHub Actions workflow to deploy the backend to Fly.io.

Steps:

1. Create a Fly account and generate an API token.
2. In your GitHub repo settings > Secrets, add `FLY_API_TOKEN` with the token value. Optionally add `FLY_APP` to choose the Fly app name (defaults to `sensro-backend`).
3. Trigger the GitHub Action `Deploy to Fly` from the Actions tab, or push to `main`.

The workflow will build the Docker image and deploy it to Fly. After deployment note the public URL and update the frontend API base (if needed).


## Tools

- `tools/export_segments.py` — fetches `/api/segments` from a running backend and writes `frontend/segments.json`.
- `tools/export_segments_from_db.py` — exports readings directly from the local `road_data.db` into `frontend/segments.json`.


## Notes

This project includes a local SQLite database and a machine learning workflow under the `ml/` directory. You may need to ensure the required model files and data sources are present before running prediction or training tasks.

## License

This project is provided as-is for educational or personal use unless otherwise specified.
