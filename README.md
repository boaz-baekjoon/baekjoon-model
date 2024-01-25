
# ![Title](https://capsule-render.vercel.app/api?type=transparent&fontColor=000000&text=백발백준%20-%20BOJ%20PS%20problem%20Recsys%20Server%20&height=200&fontSize=35&desc=BOAZ%2019th%20Big%20Data%20Conference%202024%20%20&descAlignY=76&descAlign=50)

## Abstract
**BOJ PS problem Recsys Server bulit with** 

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/>
<img src="https://img.shields.io/badge/DuckDB-FFF000?style=flat&logo=DuckDB&logoColor=white"/>

## Introduction
This GitHub repository contains the source code for BOJ problem recommendation system server, which provides APIs that the [baekjoon Bot](https://github.com/boaz-baekjoon/baekjoon-discord-bot) server can request in various user needs situations. The recommendation was implemented through a sequential recommendation model and a hybrid graph model combining Collaborative Filtering (CF) and Knowledge Graph (KG). These AI models were built with Pytorch and served with FastAPI. In addition, appropriate transformation was performed on the [pre-loaded PostgreSQL](https://github.com/boaz-baekjoon/baekjoon-celery-scraper-airflow) data and a data mart for the model server was built using duckdb.

## Data
Used the problem solving status and problem information data of silver level or higher users crawled from [BOJ](https://www.acmicpc.net/).
- user number : about 110,000 people
- problum number : about 30000
- interaction : about 17 million

## Recommendation Model

### SASRec
- SASRec is a classically used model in the field of sequential recommendation.
- Chose SASRec as a personalized recommendation model due to its parallel processing capability, efficiency in space complexity, and fast inference time.
- [The paper author's legacy tensorflow code](https://github.com/kang205/SASRec) was rewritten in pytorch.
- The default values ​from the paper were used as hyperparameters.


### KGAT
- KGAT is a hybrid graph model that models Collaborative Filtering information and side information as CF graph and knowledge graph, respectively.
- Used KGAT to generate item embeddings by appropriately using CF information and side information.
- Some typos and unnecessary operations were corrected in [the existing author's code](https://github.com/xiangwang1223/knowledge_graph_attention_network).
- For most hyperparameters, the default values ​​from the paper were used. However, we reduce the embedding dimension and number of layers 
- As a result of the experiment, when the embedding dimension was low, loss was reduced better. Perhaps the recommendation problem we are trying to solve is expected to be at a low dimension.

## Recommendation System
Figure describing overall system design of the recommendation system. 

![Alt text](image.png)
## how to use

### 🖥️ Running the server locally
```
cd baekjoon-model
uvicorn server:app --host 0.0.0.0 --port {PORTNUM} --reload
```

### 📄 endpoints docs

|endpoint | method | Model | explanation | Request | Response |
|--------|----|------|----------|------|------|
|baekjun/user_id |POST|SASRec|Recommend problems based on the user's history of problems solved in the past.|{<br>”user_id_list”: List[str],<br>”problem_num”:int<br>}|{<br>”{user_id1}”:[problems_list],<br>”{user_id2}”:<br>[problems_list]<br>...}
|baekjun/category |POST|SASRec|Recommend problems of the problem type selected by user.|{<br>”user_id_list”: str,<br>”category”:int<br>”problem_num”:int<br>}|{<br>”user_id” : List[int]<br>}|
|baekjun/group_rec |POST|SASRec|Recommend problems of the tier and problem type selected by group users.|{<br>”user_id_list” : List[str],<br>”tier” : int,<br>”category_num” : List[int]<br>}| {<br>”0” : List[int],<br>”1” : List[int],<br>…,<br>”9” List[int]<br>}|
|baekjun/similar_id |POST|KGAT|Recommend problems similar to the problem submitted by the user.|{<br>”problem_id” : int,<br>”problem_num” : int<br>}|{<br>”problem_id” : List[int]<br>}|

