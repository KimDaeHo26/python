# sqlite3   심플한 데이터베이스

<pre>
파일단위로 데이터베이스 생성됨
일반 Database 처럼 lock 처리가 아닌 파일단위의 lock 처리로 동시여러접속 불가 (insert, update, select) 
timeout 을 적은 숫자로 셋팅해서 lock 발생 최소화 하게 처리 함
</pre>

**아래 링크를 통해 주피터 노트북 뷰어(nbviewer.jupyter.org)로 보거나 구글 코랩(colab.research.google.com)에서 실행할 수 있음**

<pre>
<a target="_blank" href="https://nbviewer.jupyter.org/github/KimDaeHo26/python/blob/main/sqlite3.ipynb"><img src="https://jupyter.org/assets/main-logo.svg" width="28" />주피터 노트북 뷰어로 보기</a>
<a target="_blank" href="https://colab.research.google.com/github/KimDaeHo26/python/blob/main/sqlite3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
</pre>


```python
# 아래 주석 풀고 실행하면 주피터 노트북에서 설치 됨
# !pip install sqlite3
```


```python
import sqlite3  # 동시접속 불가
print('버전 : =>', sqlite3.version)
from os import path

dbFileNm = "./test.db"
# .py 파일 위치일때는 아래 주석 풀고 실행
# dbFileNm = path.join(path.dirname(__file__), "test.db")

# 데이터베이스 파일이 있으면 연결, 없으면 생성
con = sqlite3.connect(dbFileNm, timeout=5) 
# 커서 선언
cur = con.cursor()                          

# testT 테이블이 있으면 삭제
cur.execute('DROP TABLE IF EXISTS testT')   

# testT 테이블 생성
cur.execute('CREATE TABLE testT ' \
            ' (col1 TEXT, col2 INTERGER, col3 DOUBLE)'
            )      

# 테이블에 데이터 넣어 줌
cur.execute('insert into testT(col1, col2, col3) ' \
            ' values ("테스트", 10, 10.5)'
            )

# 커밋처리함 ( 데이터베이스 에 적용 )
con.commit()   

# 조회 쿼리
sql = "select * from testT"
# 조회 실행
cur.execute(sql)
# 결과 전체를 results 에 넣어 줌
results = cur.fetchall()

print('조회 결과 : => \n', results)

## pandas 를 이용한 데이터베이스 적재 및 조회
import pandas as pd
# 조회 쿼리
sql1 = "select * from testT"
df = pd.read_sql(sql=sql1,con=con)
print('pandas 를 이용한 조회 결과 : => \n', df)

# # pandas 데이터프래임 db 적재
# df.to_sql(name='testT',  # 'testT' 테이블에 적재
#           con=con,       # 데이터베이스 연결
#           index=False,    
#           if_exists='append') # if_exists 옵션 : append 추가, fail 테이블존재할때 아무것도 안함, replace 테이블 새로만들어서 넣음 


# pandas 데이터프래임 db 적재
df.to_sql(name="test2T",   # 'test2T' 테이블에 적재
          con=con,         # 데이터베이스 연결
          index=False, 
          if_exists='replace') # if_exists 옵션 : append 추가, fail 테이블존재할때 아무것도 안함, replace 테이블 새로만들어서 넣음 

sql = "select * from test2T"
df2 = pd.read_sql(sql=sql,con=con)
print('test2T 조회 결과 : =>\n', df2)

# 커밋처리함 ( 데이터베이스 에 적용 )
con.commit()
# 연결은 닫음 ( 안닫으면 lock 발생 : timeout 될때까지 기다려야됨 )
con.close()
```

    버전 : => 2.6.0
    조회 결과 : => 
     [('테스트', 10, 10.5)]
    pandas 를 이용한 조회 결과 : => 
       col1  col2  col3
    0  테스트    10  10.5
    test2T 조회 결과 : =>
       col1  col2  col3
    0  테스트    10  10.5
    
