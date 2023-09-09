import json
import pandas as pd
from typing import List, Optional


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import data
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

# Post Example
# @app.post("/api/summary")
# async def summary(text: Text):
#     response = get_summary(text.text)
#     return {"result": response}

