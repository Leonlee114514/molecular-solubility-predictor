"""DisSolve FastAPI application entrypoint.

Run with:
    venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from dotenv import load_dotenv

# Load KIMI_API_KEY etc. from the project-root .env (read-only; no Streamlit).
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router

# React dev server (Vite) and Kimi Work preview origin.
_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:7100",
]


def create_app() -> FastAPI:
    app = FastAPI(title="DisSolve API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()
