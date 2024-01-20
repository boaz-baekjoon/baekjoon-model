from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from endpoints import recsys_router
from init_model import args, infer_model

# App Setting Section
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(recsys_router.router)
app.include_router(preprocess_router.router)