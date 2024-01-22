from fastapi import APIRouter, Depends

from query import *
from schema.schema import UserIDRequest, CategoryRequest, SimilarIDRequest, GroupRequest
from init_model import infer_model, args

from utils.log import make_logger
from ast import literal_eval

import random

user_id_logger = make_logger("user_id_logger")

router = APIRouter(
    prefix="/baekjun"
)

@router.post("/user_id")
async def get_user(input : UserIDRequest):
    userid_problem_list = dict()
    for user_id in input.user_id_list:
        try: 
            # map user_id to user_id_int
            user_id_int = query_user_id_map(user_id)
            if user_id_int == None:
                raise Exception('user_id_int is None')

            # load user_seq
            user_seq = literal_eval(query_user_seq(user_id_int))
            if user_seq == None:
                raise Exception('user_seq is None')
            
            # load user_tier
            user_tier = query_user_tier(user_id)
            if user_tier == None:
                raise Exception('user_tier is None')
            
            # load problem_list
            problem_list = query_problem_list(user_tier-4, user_tier+4, None)
            if problem_list == None:
                raise Exception('problem_list is None')         
            problem_list = list(set(problem_list) - set(user_seq))
            
            result = infer_model.predict_for_user_sq(
                sequence=user_seq, 
                item_num=input.problem_num, 
                problem_list=problem_list,
                args=args
            )
            userid_problem_list[user_id] = list(map(int, result))
            user_id_logger.info({f"'type': 'recsys', 'user_id': '{user_id}', 'problem_list': {result}"})
            
        except: 
            problem_list = query_problem_list(4, 10, None)
            result = random.sample(problem_list, input.problem_num)
            userid_problem_list[user_id] = [problem for problem in result]
            user_id_logger.info({f"'type': 'except', 'user_id': '{user_id}', 'problem_list': {userid_problem_list[user_id]}"})
            
    return userid_problem_list

@router.post("/category")
async def get_category(input : CategoryRequest):
    userid_problem_list = dict()
    user_id = input.user_id
    try: 
        # map user_id to user_id_int
        user_id_int = query_user_id_map(user_id)
        if user_id_int == None:
            raise Exception('user_id_int is None')

        # load user_seq
        user_seq = literal_eval(query_user_seq(user_id_int))
        if user_seq == None:
            raise Exception('user_seq is None')
        
        # load problem_list
        problem_list = query_problem_list(0, 31, input.category)
        problem_list = list(set(problem_list) - set(user_seq))
        
        result = infer_model.predict_for_user_sq(
            sequence=user_seq, 
            item_num=input.problem_num, 
            problem_list=problem_list ,
            args=args
        )
        userid_problem_list[user_id] = list(map(int, result))
        user_id_logger.info({f"'type': 'recsys', 'user_id': '{user_id}', 'problem_list': {result}"})
        
    except: 
        problem_list = query_problem_list(4, 10, input.category, True)
        result = random.sample(problem_list, input.problem_num)
        userid_problem_list[user_id] = [problem for problem in result]
        user_id_logger.info({f"'type': 'except', 'user_id': '{user_id}', 'problem_list': {userid_problem_list[user_id]}"})
        
    return userid_problem_list

@router.post("/group_rec")
async def get_group_rec(input : GroupRequest):
    group_problem_list = dict()
    user_num = len(input.user_id_list)
    
    # load sequence
    each_user_seq = []
    all_user_seq = []
    for user_id in input.user_id_list:
        try:
            # map user_id to user_id_int
            user_id_int = query_user_id_map(user_id)
            if user_id_int == None:
                raise Exception('user_id_int is None')

            # load user_seq
            user_seq = literal_eval(query_user_seq(user_id_int))
            if user_seq == None:
                raise Exception('user_seq is None')
            
            each_user_seq.append(user_seq)
            all_user_seq.extend(user_seq)
        except:
            user_seq = []
            each_user_seq.append(user_seq)

    cat = 0
    for cat_num in input.category_num:
        print("class: ", cat)
        if cat_num == 0:
            group_problem_list[cat] = []
            cat +=1
            continue
        problem_list = query_problem_list(input.tier-2, input.tier+2, cat, True)
        try:
            problem_list = random.sample(problem_list, 100)
        except:
            problem_list = random.sample(problem_list, len(problem_list))
        problem_list = list(set(problem_list) - set(all_user_seq))
        
        # predict
        predict_array = np.zeros((user_num, len(problem_list)))
        
        for user_idx in range(user_num):
            if len(each_user_seq[user_idx]) == 0:
                predict_array[user_idx] = np.zeros(len(problem_list))
                continue
            predict = infer_model.return_prob(
                sequence=each_user_seq[user_idx], 
                problem_list=problem_list,
                args=args
            )
            predict_array[user_idx] = predict
        print(len(problem_list))
        print(predict_array.mean(axis=0).shape)
        group_problem_list[cat] = [problem_list[idx] for idx in list(np.argsort(predict_array.mean(axis=0))[-cat_num:])]     
        cat += 1   
    return group_problem_list
        

@router.post("/similar_id")
async def get_problem_id(input : SimilarIDRequest):
    if input.problem_id not in unique_problem_list:
        return {'error': 'problem_id is not in problem_list'}
    top_n = 11
    problem_id_dict = dict()
    similar_problem_list = cosine_sim[input.problem_id].argsort()[::-1][1:top_n].tolist()
    while True:
        if len(set(similar_problem_list) & set(unique_problem_list)) >= input.problem_num:
            break
        top_n += 10
        print(top_n)
        similar_problem_list = cosine_sim[input.problem_id].argsort()[::-1][1:top_n].tolist()
    similar_problem_list = list(set(similar_problem_list) & set(unique_problem_list))
    problem_id_dict[input.problem_id] = np.random.choice(similar_problem_list, input.problem_num, replace=False).tolist()
    return problem_id_dict

# @router.get("/similar_text")
# async def get_problem_id(problem_text : str):
#     similar_problem_list = dict()
#     result = list(problem_list.sample(n=3).to_dict(orient='records'))
#     similar_problem_list['problems'] = [problem['problem_id'] for problem in result]
#     return similar_problem_list

