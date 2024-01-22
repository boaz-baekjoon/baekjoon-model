from pydantic import BaseModel
from typing import List

class RandomRequest(BaseModel):
    user_id : str
    prob_num : int

class UserIDRequest(BaseModel):
    user_id_list : List[str]
    problem_num : int
    
class CategoryRequest(BaseModel):
    user_id : str
    category : int
    problem_num : int
    
class GroupRequest(BaseModel):
    user_id_list : List[str]
    tier : int
    category_num : List[int] # [implementation, ds, dp, graph, search, string, math, opt, geo, adv]
    
class SimilarIDRequest(BaseModel):
    problem_id : int
    problem_num : int