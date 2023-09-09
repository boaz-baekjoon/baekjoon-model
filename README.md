## Model Server

### 📦 project structure
```
baekjoon-model
├─ schema
│  └─ schema.py
├─ endpoints
│  ├─ recsys_router.py 
│  ├─ io_router.py 
│  └─ preprocess_router.py 
├─ model
│  ├─ train.py
│  ├─ test.py
│  ├─ sasrec
│  │  ├─ module
│  │  │  └─ sample_module.py
│  │  ├─ data_loader.py
│  │  ├─ loss.py
│  │  └─ model.py
│  ├─ utils
│  │  └─ sample_utils.py
│  └─ results
├─ data_preprocessing
│  └─ preprocessing.py
├─ data
├─ utils
├─ server.py
├─ database.py
├─ .gitignore
└─ README.md
```

### ✍️ Commit Message Convention

**Types**
```
- Feat : 기능 추가
- Chore : 기타 수정
- Fix : 버그 수정
- Docs : 문서 수정
- Dev : dependency 수정
- Test : 테스트 코드, 리팩토링 테스트 코드 추가
- Comment : 필요한 주석 추가 및 변경
- Rename : 파일 또는 폴더 명을 수정하거나 옮기는 작업만인 경우
- Remove : 파일을 삭제하는 작업만 수행한 경우
- Style : 코드 formatting, 세미콜론 누락, 코드 자체의 변경이 없는 경우
- Refactor : 코드 리팩토링
- !BREAKING CHANGE : 커다란 API 변경의 경우
- !HOTFIX : 급하게 치명적인 버그를 고쳐야 하는 경우
```

**Issue Labels**
```
- Feat: 기능 추가
- Chore: 코드 정리나 주석 추가 등 구현과 직접적으로 관련이 없는 내용
- Docs: README 등의 문서화
- Fix: 버그 수정 또는 예외처리
- Experiment : model log, weight, modification, ...
```

**Message**
- 커밋 유형과 이슈 번호 명시
    - git commit -m "[커밋 유형] #[이슈 번호] [커밋메시지]"
- 제목과 본문을 빈행을 분리
    - 커밋 유형 이후 제목과 본문은 한글로 작성하여 내용이 잘 전달될 수 있도록
    - 본문에는 변경한 내용과 이유 설명
- 제목 첫 글자는 대문자로 끝에 . 금지
- 제목은 영문 기준 50자 이내로 작성

**Message Examples**
```
[Feat] Add data preprocessing code
[Fix] Fix bugs
[Docs] Update .gitignore
[Comment] Add comments
```