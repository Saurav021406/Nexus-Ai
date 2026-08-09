from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health
from app.routers import upload
from app.routers import domain
from app.routers import chat
from app.routers import forecast
from app.routers import cleaning
from app.routers import eda
from app.routers import viz

app = FastAPI(title="Nexus AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(domain.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(forecast.router)
app.include_router(cleaning.router)
app.include_router(eda.router)
app.include_router(viz.router)


@app.get("/")
async def root():
    return {"message": "Nexus AI backend is running. See /docs for API docs."}
