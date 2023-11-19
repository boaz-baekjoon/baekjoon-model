from fastapi import APIRouter, Depends

from database import *
from schema.schema import UserIDRequest
from init_model import infer_model, args

from utils.log import make_logger
from ast import literal_eval

user_id_logger = make_logger("user_id_logger")

router = APIRouter(
    prefix="/baekjun"
)

@router.post("/user_id")
async def get_user(input : UserIDRequest):
    userid_problem_list = dict()
    for user_id in input.user_id_list:
        try: 
            user_id_int = user_id_dict[user_id]
            user_problem_list = user_problem_table[user_problem_table.user_id_int == user_id_int]
            user_seq = literal_eval(user_problem_list.iloc[0, 1])
            user_tier = user_detail[user_detail.user_id == user_id].iloc[0, 1]
            
            problem_list = problem_detail[(problem_detail.problem_level >= user_tier - 3) & (problem_detail.problem_level >= user_tier + 3)]
            result = infer_model.predict_for_user_sq(
                sequence=user_seq, 
                item_num=input.problem_num, 
                problem_list=problem_list ,
                args=args
            )
            userid_problem_list[user_id] = list(map(int, result))
            user_id_logger.info({f"'type': 'recsys', 'user_id': '{user_id}', 'problem_list': {result}"})
            
        except: 
            problem_list = problem_detail[(problem_detail.problem_level <= 15)][['problem_id']]
            result = list(problem_list.sample(n=input.problem_num).to_dict(orient='records'))
            userid_problem_list[user_id] = [problem['problem_id'] for problem in result]
            
            user_id_logger.info({f"'type': 'except', 'user_id': '{user_id}', 'problem_list': {userid_problem_list[user_id]}"})

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

