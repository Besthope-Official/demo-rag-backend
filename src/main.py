"""FastAPI 应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pymongo import MongoClient

from src.cache import LocalCache
from src.core.api import router as core_router
from src.database import MongoConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # config = RedisConfig.from_yaml()
        app.state.cache = LocalCache()
        # app.state.cache = RedisCache(config=config)
    except Exception as e:
        logger.warning(f"Failed to initialize RedisCache: {e}, falling back to LocalCache")
        app.state.cache = LocalCache()
    try:
        mongo_cfg = MongoConfig.from_yaml()
        mongo_client = MongoClient(mongo_cfg.url)

        app.state.mongo_client = mongo_client
        app.state.mongo_db = mongo_client[mongo_cfg.database]
        yield
    finally:
        # sync API
        app.state.mongo_client.close()
        await app.state.cache.close()


app = FastAPI(title="demo-rag-backend", lifespan=lifespan)

# Allow ALL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(core_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
