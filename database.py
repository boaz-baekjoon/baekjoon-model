import pandas as pd

# [columns] problem_id,problem_title,problem_info,problem_answer_num,problem_submit_num,problem_answer_rate
# [row 1] 1000,A+B,다국어,257995,925751,39.953
# [row 2] 1001,A-B,,220129,378852,70.162
# [row 3] 1002,터렛,,34142,200019,22.403
# [row 4] 1003,피보나치 함수,,47315,199262,32.666
# [row 5] 1004,어린 왕자,,14065,37494,45.718
data = pd.read_csv("data/preproc_data/problem_id.csv")
