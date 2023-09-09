import json
from fastapi import APIRouter

from database import data

router = APIRouter(
    prefix="/api"
)

# input : user_id(str), 문제 개수(int), 
# output : problem_number_list(json) - 입력받은 문제 개수 만큼 문제 번호 list를 생성해서 json으로 반환
# ex) {problems : { 0 : 123, 1 : 13, 2 : 14}} 
@router.get("/random")
async def get_random(user_id: str, num: int):
    result = data.sample(n=num).to_dict(orient='records')
    result = { i : problem["problem_id"] for i, problem in enumerate(result) }
    return json.dumps({"problems": result})
