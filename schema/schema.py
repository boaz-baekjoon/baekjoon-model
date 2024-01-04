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