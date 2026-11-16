# Deployment

The service is a single container that serves the API and the web interface on one port. The trained
model (`artifacts/model.joblib`, under 1 MB) is copied into the image, so no training happens at build
time and the container needs no database or external service.

## Build and run locally

```bash
docker build -t credit-risk:latest .
docker run --rm -p 8000:8000 credit-risk:latest
```

Open `http://localhost:8000/` for the interface and `http://localhost:8000/docs` for the API
documentation. `docker compose up --build` does the same using `docker-compose.yml`.

## Deploy to a container hosting service

The steps are the same for most container platforms (managed container services, platform-as-a-service
offerings that run Docker images, or a small virtual machine with Docker).

1. Build the image in CI or locally and push it to a container registry the platform can read:

   ```bash
   docker tag credit-risk:latest <registry>/<namespace>/credit-risk:1.0.0
   docker push <registry>/<namespace>/credit-risk:1.0.0
   ```

2. Create a web service from that image. Expose the container port (default `8000`).
3. If the platform assigns the port through a `PORT` variable, no change is needed: the container
   listens on `$PORT` and defaults to 8000.
4. Set the environment variables below, if any differ from the defaults.
5. Configure the health check to call `GET /health` and expect HTTP 200 with `"status": "ok"`.
6. Put the service behind HTTPS (most platforms terminate TLS for you) and restrict `CORS_ORIGINS`
   to the origins that call the API from a browser. The bundled interface is served from the same
   origin and needs no CORS setting.

## Environment variables

All variables are optional and read at startup.

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port the server listens on inside the container |
| `CREDIT_RISK_MODEL_PATH` | `/app/artifacts/model.joblib` | Fitted model file |
| `CREDIT_RISK_MODEL_CARD_PATH` | `/app/artifacts/model_card.json` | Model card (threshold, metrics, schema) |
| `CREDIT_RISK_CORS_ORIGINS` | `[]` | JSON list of allowed origins, for example `["https://example.org"]` |
| `CREDIT_RISK_MAX_BATCH_SIZE` | `100` | Maximum applicants per batch request |
| `CREDIT_RISK_MAX_BODY_BYTES` | `256000` | Maximum request body size |
| `CREDIT_RISK_LOG_LEVEL` | `INFO` | Log level; logs are JSON lines without request bodies |
| `CREDIT_RISK_DOCS_ENABLED` | `true` | Serve `/docs` and `/openapi.json` |
| `CREDIT_RISK_EXPLANATION_TOP_K` | `5` | Default number of contributing features |

## Operational notes

- Free tiers of container hosting services commonly stop an idle service and restart it on the next
  request. The first request after a pause can take several seconds while the model loads; the
  `/health` endpoint reports `degraded` until it is ready.
- The service is stateless and keeps no user data; scale by running more replicas.
- The image runs as a non-root user and ships a Docker health check.
- Retrain by running `make train` and `make evaluate` and rebuilding the image so that the new
  `artifacts/` are included. Keep the library versions in `requirements-serve.txt` identical to the
  ones recorded in `artifacts/model_card.json`, because the model file is a pickled scikit-learn and
  LightGBM object.
- The model is an educational demonstration on 2005 Taiwanese data and must not be used for real
  lending decisions.
