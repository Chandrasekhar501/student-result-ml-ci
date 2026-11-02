# Student Result Prediction - MLOps Practical Series

This repository implements the Student Result Prediction project and the practical progression for ML CI/CD, model quality validation, artifact preservation, API delivery, Docker, GHCR, cloud deployment preparation, and application logging.

## Project files

- `train_model.py`: creates a reproducible 300-record dataset, trains Logistic Regression, and writes the model and metrics.
- `test_ml_pipeline.py`: automated model and artifact tests.
- `quality_gate.py`: requires model accuracy of at least 0.85.
- `flask_app.py` / `test_app.py`: Flask prediction API used for the earlier API and Docker practical stages.
- `app.py` / `test_api.py`: FastAPI service used for the later GHCR and Render practical stages. Includes request IDs, timing, validation and prediction logs.
- `Dockerfile`: packages the FastAPI service and trained model on port 8000.
- `.github/workflows/ml-ci.yml`: training, quality gate, tests, artifact uploads, release candidate, Docker build and container smoke test.
- `.github/workflows/docker-publish.yml`: tests and publishes the Docker image to GHCR after successful validation.

## Run locally

```bash
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI.

## Prediction request

POST `/predict`

```json
{
  "attendance": 90,
  "internal_marks": 85,
  "assignment_marks": 88,
  "previous_score": 80
}
```

## GitHub Actions and artifacts

Push to `main` or use Actions > Run workflow. The ML CI workflow uploads:
- `student-result-ml-artifacts`
- `student-result-ml-release`

The GHCR workflow publishes `ghcr.io/chandrasekhar501/student-result-prediction` with `latest` and commit-SHA tags when the repository has package-write permission.

## Render deployment

After the GHCR workflow succeeds, create a Render Web Service from the GHCR image. Set `PORT=8000` and health-check path `/health`. Render account setup, any private-registry credential, and the actual deployment must be completed in the Render dashboard. Never commit registry tokens or secrets.
