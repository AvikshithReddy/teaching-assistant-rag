import sys
from pathlib import Path

# Ensure backend/ is on the path so routers can import auth, main_app, app.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.auth import router as auth_router
from routers.courses import router as courses_router
from routers.materials import router as materials_router
from routers.index import router as index_router
from routers.chat import router as chat_router

app = FastAPI(title="Teaching Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(courses_router, prefix="/api")
app.include_router(materials_router, prefix="/api")
app.include_router(index_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
