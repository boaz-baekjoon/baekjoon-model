import json
from fastapi import APIRouter, Depends
from typing import List

from database import problem_list
from schema.schema import UserIDRequest


router = APIRouter(
    prefix="/baekjun"
)

@router.post("/user_id")
async def get_user(input : UserIDRequest):
    userid_problem_list = dict()
    for user_id in input.user_id_list:
        result = list(problem_list.sample(n=input.problem_num).to_dict(orient='records'))
        userid_problem_list[user_id] = [problem['problem_id'] for problem in result]
    return userid_problem_list

@router.get("/similar_id")
async def get_problem_id(problem_id : int):
    similar_problem_list = dict()
    result = list(problem_list.sample(n=3).to_dict(orient='records'))
    similar_problem_list['problems'] = [problem['problem_id'] for problem in result]
    return similar_problem_list

@router.get("/similar_text")
async def get_problem_id(problem_text : str):
    similar_problem_list = dict()
    result = list(problem_list.sample(n=3).to_dict(orient='records'))
    similar_problem_list['problems'] = [problem['problem_id'] for problem in result]
    return similar_problem_list