# python
## python 3.8 버전 이용

# 엑셀파일 자동 생성 프로그램

<pre>
엑셀프로그램을 구동 시켜 엑셀의 메크로 함수를 호출하여 
tuple, list, pandas.Dataframe, numpy.ndarray, image파일, matplotlib.pyplot 을 넣어주는 프로그램임

사용환경 : windows, MS_Office 설치된 환경
설치방법 : pip install KDHexcel
</pre>

```python
from KDHexcel import KDHexcel

x = KDHexcel.KDHexcel()

print("버전:", x.__버전__)
print("작성자:", x.__작성자__)

help(x.셀값넣기)
help(x.이미지파일넣기)
help(x.그래프넣기)
help(x.셀값지우기)

# 첫번째 시트에 1개의 값 넣어줌
x.셀값넣기(셀="A1", 값="첫번째시트")

# 테스트 시트에 1개의 값 넣어줌
x.셀값넣기(셀="A1", 값="abcdef", 시트명="테스트")
x.셀값넣기("B1",5.15,"테스트")

# 튜플 시트에 tuple 데이터 넣어줌
t1 = ("튜플a","튜플b","튜플c","튜플d")
x.셀값넣기("B3",t1,"튜플")

# 리스트11 시트에 list 데이터 넣어줌
l1 = ["리스트1","리스트2","리스트3"]
x.셀값넣기("A3",l1,"리스트11")

# 리스트2 시트에 2차원 list 데이터 넣어줌
l2 = [ ["리스트21","리스트22","리스트23"],
       ["리스트1","리스트2","리스트3"],
     ]
x.셀값넣기("A1",l2,"리스트2")

# pandas 데이터셋 넣어줌
import pandas as pd
a = [["a1","b1"],["c1","d1"]]
p1 = pd.DataFrame(a, index=list('ab'), columns=list('de'))
x.셀값넣기("A1",p1,"판다스")

# numpy array 넣어줌
import numpy as np
lst1 = [1, 2, 3, 4, 5, 6]
넘파이 = np.array(lst1)
x.셀값넣기("a1",넘파이,"넘파이")

lst2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
넘파이 = np.array(lst2)
x.셀값넣기("b3",넘파이,"넘파이")

# 이미지파일 넣기
x.이미지파일넣기("d4", 시트명="이미지", 파일명=r"C:\Users\User\파이썬주피터\plot3.png",ColumnWidth=50, RowHeight=150)

# matplotlib.pyplot 넣기
import matplotlib.pyplot as plt
# 샘플그래프 그리기 시작
import numpy as np
a = np.random.normal(size=50)
plt.hist(a, bins=5)
# 샘플그래프 그리기 끝

x.그래프넣기("f8",plt,ColumnWidth=30, RowHeight= 130, 시트명="그래프")

a = [[1,2],[2,4],[3,3]]
p1 = pd.DataFrame(a, index=list('abc'), columns=list('de'))
p1.plot.line(x='d',y='e')

x.그래프넣기("a1",plt,ColumnWidth=30, RowHeight= 130, 시트명="그래프")

x.셀값넣기(셀="A1", 값=[1,2,3,4,5,6], 시트명="셀값지우기")
x.셀값지우기(셀="B1:D1", 시트명="셀값지우기")
```

    버전: 1.0
    작성자: 김대호
    Help on method 셀값넣기 in module KDHexcel.KDHexcel:
    
    셀값넣기(셀='A1', 값='1', 시트명=None) method of KDHexcel.KDHexcel.KDHexcel instance
        :param 셀 : 저장될 셀위치 (예 : B3)
        :param 값 : 셀에 들어갈 값 (1개의 값 또는 tuple, list, pandas.Dataframe, numpy.ndarray)
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :return: 없음
    
    Help on method 이미지파일넣기 in module KDHexcel.KDHexcel:
    
    이미지파일넣기(셀, 파일명, ColumnWidth=50, RowHeight=150, 시트명=None) method of KDHexcel.KDHexcel.KDHexcel instance
        :param 셀 : 저장될 셀위치 (예 : B3)
        :param 파일명: 이미지 파일명 (예 : 'C:\Users\User\파이썬주피터\plot3.png' )
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :param ColumnWidth : 셀의 너비
        :param RowHeight : 셀의 높이
        :return: 없음
    
    Help on method 그래프넣기 in module KDHexcel.KDHexcel:
    
    그래프넣기(셀, plt, ColumnWidth=50, RowHeight=150, 시트명=None) method of KDHexcel.KDHexcel.KDHexcel instance
        :param 셀: 저장될 셀위치 (예 : B3)
        :param plt: 그래프 object (예 :  matplotlib.pyplot )
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :param ColumnWidth : 셀의 너비
        :param RowHeight : 셀의 높이
        :return: 없음
    
    Help on method 셀값지우기 in module KDHexcel.KDHexcel:
    
    셀값지우기(셀='A1', 시트명=None) method of KDHexcel.KDHexcel.KDHexcel instance
        :param 셀 : 지울 셀위치 (예 : B3, A1:B3)
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :return: 없음
    
    




# 재무비율계산
## jemu.csv 파일 : 재무비율계산 용 재무제표 샘플


*    KDHcalcClass.calc(upche, fmul, dateC, dateB, dateA, option: str = 'y')
*    """ 산식(&00-0001C&+...+&00-0005C&)을 금액으로 치환해서 계산 결과를 리턴해줌
*    ex) (&11-0001C&)+&12-0001C&
*        -> 100+&12-0001C&
*        -> 100+200
*        -> 300 리턴
*
*    산식예외처리
*    math library를 이용 
*        log,exp,log10,pow,sqrt 계산
*    별도 함수를 만들어 처리 
*        if, or, and
*    등호를 python 문법으로 치환
*        = -> ==
*
*    최종계산 eval 함수 이용
