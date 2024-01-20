from fastapi import APIRouter
import pandas as pd
import numpy as np

from utils.seq_preproc import *
from utils.problem_preproc import *
from utils.user_preproc import *

router = APIRouter(
    prefix="/baekjun"
)

@router.get("/load_and_preproc_seq")
async def load_seq():
    print("Start loading user sequence data")
    load_seq_data()
    print("Start preprocessing user sequence data")
    preproc_seq()
    return {"message": "success"}

@router.get("/load_and_preproc_problem")
async def load_problem():
    print("Start loading problem data")
    load_problem_data()
    print("Start preprocessing problem data")
    preproc_problem()
    return {"message": "success"}

@router.get("/load_user")
async def load_user():
    print("Start loading user data")
    load_user_data()
    return {"message": "success"}
