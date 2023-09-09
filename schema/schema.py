from pydantic import BaseModel

class RandomRequest(BaseModel):
    user_id : str
    prob_num : int
