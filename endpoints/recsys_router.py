import json
from fastapi import APIRouter, Depends
from typing import List

from database import *
from schema.schema import UserIDRequest

from init_model import infer_model, args

router = APIRouter(
    prefix="/baekjun"
)

# load model
@router.post("/user_id")
async def get_user(input : UserIDRequest):
    userid_problem_list = dict()
    for user_id in input.user_id_list:
        try: 
            user_id_int = user_id_dict[user_id]
            user_problem_list = user_problem_table[user_problem_table.user_id_int == user_id_int]
            user_seq = [int(x[1:-1])for x in user_problem_list.iloc[0,1].strip("[]").split(", ")]
            result = infer_model.predict_for_user_sq(user_seq, input.problem_num, problem_list ,args)
            userid_problem_list[user_id] = result 
        except:
            # todo : solve.ac 에서 유저 정보 가져오기
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

