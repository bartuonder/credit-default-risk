import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from credit_risk import __version__
from credit_risk.api.middleware import (
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    configure_logging,
)
from credit_risk.api.schemas import (
    EXAMPLE_HIGH_RISK,
    EXAMPLE_LOW_RISK,
    Applicant,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    Health,
    PredictionResponse,
    to_frame,
)
from credit_risk.api.service import ModelService
from credit_risk.api.settings import Settings

log = logging.getLogger("credit_risk.api")
STATIC_DIR = Path(__file__).parent / "static"

DESCRIPTION = """
Probability of credit card default in the next month, from the UCI *Default of Credit Card Clients*
data (Taiwan, 2005).

**Educational demonstration. Not for real lending decisions.**

Probabilities are calibrated. The decision flag applies a cost-based threshold that assumes a missed
default costs more than a declined good client; the ratio is reported by `/model-info`.
"""

ERRORS = {
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
}


def error_response(
    status: int, code: str, message: str, details: list | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or []}},
    )


def _field_path(loc: tuple) -> str:
    return ".".join(str(p) for p in loc if p not in ("body", "query"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        try:
            app.state.service = ModelService(settings.model_path, settings.model_card_path)
        except FileNotFoundError:
            log.error("model artifacts not found; predictions are unavailable")
        yield

    app = FastAPI(
        title="Credit default risk API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        openapi_tags=[
            {"name": "service", "description": "Health and model metadata."},
            {"name": "prediction", "description": "Default probability, decision and explanation."},
        ],
    )
    app.state.settings = settings
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        details = [
            {"field": _field_path(e["loc"]) or None, "message": e["msg"]} for e in exc.errors()
        ]
        return error_response(422, "validation_error", "The request is invalid.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}.get(
            exc.status_code, "http_error"
        )
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception):
        log.exception("unhandled error", exc_info=exc)
        return error_response(500, "internal_error", "Internal server error.")

    def service(request: Request) -> ModelService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(503, "Model is not loaded.")
        return svc

    @app.get(
        "/health", response_model=Health, tags=["service"], summary="Liveness and model status"
    )
    async def health(request: Request):
        loaded = request.app.state.service is not None
        return Health(
            status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__
        )

    @app.get(
        "/model-info",
        tags=["service"],
        summary="Metrics, threshold, cost assumption and calibration data",
        responses=ERRORS,
    )
    async def model_info(request: Request):
        return service(request).info()

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        response_model_exclude_none=True,
        tags=["prediction"],
        summary="Score one client",
        responses=ERRORS,
    )
    async def predict(
        request: Request,
        applicant: Annotated[
            Applicant,
            Body(
                openapi_examples={
                    "low_risk": {"summary": "Pays in full every month", "value": EXAMPLE_LOW_RISK},
                    "high_risk": {"summary": "Repeated payment delays", "value": EXAMPLE_HIGH_RISK},
                }
            ),
        ],
        explain: Annotated[
            bool, Query(description="Include the top contributing features")
        ] = False,
        top_k: Annotated[
            int, Query(ge=1, le=15, description="Number of contributions")
        ] = settings.explanation_top_k,
    ):
        svc = service(request)
        result = svc.score(to_frame([applicant]), explain, top_k)[0]
        return {
            **result,
            "model_version": svc.version,
            "explanation_unit": svc.explanation_unit if explain else None,
        }

    @app.post(
        "/predict/batch",
        response_model=BatchResponse,
        response_model_exclude_none=True,
        tags=["prediction"],
        summary="Score several clients",
        description=f"At most {settings.max_batch_size} clients per request.",
        responses=ERRORS,
    )
    async def predict_batch(
        request: Request,
        body: BatchRequest,
        explain: Annotated[bool, Query()] = False,
        top_k: Annotated[int, Query(ge=1, le=15)] = settings.explanation_top_k,
    ):
        svc = service(request)
        if len(body.applicants) > settings.max_batch_size:
            return error_response(
                422,
                "batch_too_large",
                f"A batch can hold at most {settings.max_batch_size} applicants.",
                [{"field": "applicants", "message": f"received {len(body.applicants)}"}],
            )
        results = svc.score(to_frame(body.applicants), explain, top_k)
        return {
            "model_version": svc.version,
            "count": len(results),
            "predictions": results,
            "explanation_unit": svc.explanation_unit if explain else None,
        }

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
