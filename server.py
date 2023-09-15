import json
import pandas as pd
from typing import List, Optional


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from endpoints import recsys_router

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
