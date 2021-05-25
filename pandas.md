**아래 링크를 통해 주피터 노트북 뷰어(nbviewer.jupyter.org)로 보거나 구글 코랩(colab.research.google.com)에서 실행할 수 있음**

<pre>
<a target="_blank" href="https://nbviewer.jupyter.org/github/KimDaeHo26/python/blob/main/pandas.ipynb"><img src="https://jupyter.org/assets/main-logo.svg" width="28" />주피터 노트북 뷰어로 보기</a>
<a target="_blank" href="https://colab.research.google.com/github/KimDaeHo26/python/blob/main/pandas.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
</pre>

# pandas


```python
import pandas as pd
print(pd.__version__)
```

    1.1.3
    

## Series


```python
# Series는 1차원 배열, Data Frame은 2차원 배열과 유사
s1 = pd.Series([5, 3, 4, 9]) # list 사용
print(s1)
print('\n values =>', s1.values)  
print('\n index =>', s1.index)  
print('\n dtypes =>', s1.dtypes)
```

    0    5
    1    3
    2    4
    3    9
    dtype: int64
    
     values => [5 3 4 9]
    
     index => RangeIndex(start=0, stop=4, step=1)
    
     dtypes => int64
    


```python
s2 = pd.Series([5, 3, 4, 9], index=['p', 'e', 's', 'v'])  # index 직접 부여
print(s2.index)
```

    Index(['p', 'e', 's', 'v'], dtype='object')
    


```python
s3 = pd.Series({'math': 95, 'lang': 80, 'phys': 100, 'chem': 90})  # 딕셔너리 이용
print(s3)
```

    math     95
    lang     80
    phys    100
    chem     90
    dtype: int64
    


```python
# 인덱스를 새롭게 부여하는 경우, 기존에 있던 인덱스는 값을 동일하게 가지지만, 새롭게 추가된 인덱스의 경우 값이 NaN으로 부여됨.
print(pd.Series(s3, index=['biol', 'comp', 'math', 'phys']))
```

    biol      NaN
    comp      NaN
    math     95.0
    phys    100.0
    dtype: float64
    


```python
# 이름부여
s3.name = 'Scores'
s3.index.name = 'Subject'
print(s3)
```

    Subject
    math     95
    lang     80
    phys    100
    chem     90
    Name: Scores, dtype: int64
    


```python
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

print(s)
print(" index 를 통해 값 가져옴 : s['a']  => ", s['a'])     # 인덱스 값(key)으로
print(" index 를 통해 값 가져옴 : s.loc['b']  => ", s.loc['b']) # 인덱스 값(key)으로
print(" index 를 통해 값 가져옴 : s.c  => ", s.c)        # 인덱스 값(key)으로
print(" 인덱스 위치를 통해 가져옴 2번째 : s[1]  => ", s[1])         # 인덱스 위치를 통해 
print(" 인덱스 위치를 통해 가져옴 3번째 : s.iloc[2]  => ", s.iloc[2])     # 인덱스 위치를 통해 엑세스
print(" 인덱스 위치를 통해 가져옴 1~2번째 : s[1:3]  => \n", s[1:3])      # 인덱스 위치를 통해 
```

    a    1
    b    2
    c    3
    d    4
    dtype: int64
     index 를 통해 값 가져옴 : s['a']  =>  1
     index 를 통해 값 가져옴 : s.loc['b']  =>  2
     index 를 통해 값 가져옴 : s.c  =>  3
     인덱스 위치를 통해 가져옴 2번째 : s[1]  =>  2
     인덱스 위치를 통해 가져옴 3번째 : s.iloc[2]  =>  3
     인덱스 위치를 통해 가져옴 1~2번째 : s[1:3]  => 
     b    2
    c    3
    dtype: int64
    


```python
#논리형 인덱싱
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print('3 이상인지?: s>=3  => \n', s>=3)
print('\n 3 이상인것만: s[s>=3]  => \n', s[s>=3])
```

    3 이상인지?: s>=3  => 
     a    False
    b    False
    c     True
    d     True
    dtype: bool
    
     3 이상인것만: s[s>=3]  => 
     c    3
    d    4
    dtype: int64
    


```python
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print('index =>', s.index)
s1 = s.reindex(['d', 'b', 'c', 'a'])
print('index 재무여 =>', s1.index)
s2 = pd.Series(s, index=['d', 'c', 'b', 'a'])
print('index 재무여 =>', s2.index)
print('순서만바뀜 =>', s.b, s1.b, s2.b)
```

    index => Index(['a', 'b', 'c', 'd'], dtype='object')
    index 재무여 => Index(['d', 'b', 'c', 'a'], dtype='object')
    index 재무여 => Index(['d', 'c', 'b', 'a'], dtype='object')
    순서만바뀜 => 2 2 2
    


```python
#NaN 값이 생기지 않게 하기 위해서 fill_value를 사용
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s1 = s.reindex(['a', 'b', 'c', 'd', 'e', 'f'])
s2 = s.reindex(['a', 'b', 'c', 'd', 'e', 'f'], fill_value=100)
print('s  =>', s.values)
print('reindex 인덱스추가 =>', s1.values)
print('reindex fill_value=100  =>', s2.values)
```

    s  => [1 2 3 4]
    reindex 인덱스추가 => [ 1.  2.  3.  4. nan nan]
    reindex fill_value=100  => [  1   2   3   4 100 100]
    


```python
s = pd.Series(['a', 'b', 'c'], index=[0, 2, 4])
s1 = s.reindex([0, 1, 2, 3, 4, 5])
s2 = s.reindex([0, 1, 2, 3, 4, 5], method='ffill') # NaN 값이 아닌 이전 값을 가지게 하기 위해 method='ffill'을 사용

print('원본 =>\n', s,)
print('\n reindex 인덱스추가 =>\n', s1)
print("\n reindex 이전값넣어줌: method='ffill'  => \n", s2) 
```

    원본 =>
     0    a
    2    b
    4    c
    dtype: object
    
     reindex 인덱스추가 =>
     0      a
    1    NaN
    2      b
    3    NaN
    4      c
    5    NaN
    dtype: object
    
     reindex 이전값넣어줌: method='ffill'  => 
     0    a
    1    a
    2    b
    3    b
    4    c
    5    c
    dtype: object
    

# pandas Dataframe


```python
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('DataFrame =>\n', df)
print('values =>\n', df.values)
print('index =>', df.index)
```

    DataFrame =>
        0  1  2
    0  1  2  3
    1  4  5  6
    2  7  8  9
    values =>
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    index => RangeIndex(start=0, stop=3, step=1)
    


```python
df = pd.DataFrame([['a', 'b', 'c'], [4, 5, 6], [7, 8, 9]])
print('원본: df: =>\n', df)
print('\n 두번째컬럼: df[1]: =>\n', df[1])
print('\n 두번째컬럼의 세번째 : df[1][2]:  =>', df[1][2])
print('\n 첫번째줄: list(df.iloc[0]) : =>', list(df.iloc[0]))
```

    원본: df: =>
        0  1  2
    0  a  b  c
    1  4  5  6
    2  7  8  9
    
     두번째컬럼: df[1]: =>
     0    b
    1    5
    2    8
    Name: 1, dtype: object
    
     두번째컬럼의 세번째 : df[1][2]:  => 8
    
     첫번째줄: list(df.iloc[0]) : => ['a', 'b', 'c']
    

## 딕셔너리 이용


```python
data = {'subject' : ['math', 'comp', 'phys', 'chem'],
       'score': [100, 90, 85, 95],
       'students': [94, 32, 83, 17]}
df = pd.DataFrame(data)
print(df)
```

      subject  score  students
    0    math    100        94
    1    comp     90        32
    2    phys     85        83
    3    chem     95        17
    


```python
print('컬럼리스트: df.columns =>', df.columns)
print('총항목개수: df.size => ', df.size)
print('차원수: df.ndim =>', df.ndim)
print('항목개수: df.shape =>', df.shape)
print('컬럼별항목개수: df.count() =>\n', df.count())
```

    컬럼리스트: df.columns => Index(['subject', 'score', 'students'], dtype='object')
    총항목개수: df.size =>  12
    차원수: df.ndim => 2
    항목개수: df.shape => (4, 3)
    컬럼별항목개수: df.count() =>
     subject     4
    score       4
    students    4
    dtype: int64
    


```python
# 요약정보 
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4 entries, 0 to 3
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   subject   4 non-null      object
     1   score     4 non-null      int64 
     2   students  4 non-null      int64 
    dtypes: int64(2), object(1)
    memory usage: 224.0+ bytes
    


```python
# 요약정보
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>students</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>92.500000</td>
      <td>56.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.454972</td>
      <td>37.722672</td>
    </tr>
    <tr>
      <th>min</th>
      <td>85.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>88.750000</td>
      <td>28.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>92.500000</td>
      <td>57.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>96.250000</td>
      <td>85.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>94.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = pd.DataFrame(df, columns=['score', 'students'])
print('2개 컬럼만: =>\n', df1)
```

    2개 컬럼만: =>
        score  students
    0    100        94
    1     90        32
    2     85        83
    3     95        17
    


```python
# 콜론 기호를 이용하여 행과 값을 동시에 지정
z = {'math': {1:94, 2:82},
    'comp': {1:48, 3:43, 2:83}}

print('콜론 기호이용 행과 값을 동시에 지정: =>\n', pd.DataFrame(z))
```

    콜론 기호이용 행과 값을 동시에 지정: =>
        math  comp
    1  94.0    48
    2  82.0    83
    3   NaN    43
    


```python
import numpy as np
a = pd.DataFrame(np.random.randint(0, 100, (6, 4)))
a.columns = ['a', 'b', 'c', 'd']
a.index = pd.date_range('20210114', periods=6)

print('numpy array 이용:=>\n', a)
```

    numpy array 이용:=>
                  a   b   c   d
    2021-01-14  19  94  71  59
    2021-01-15  71   9  68  91
    2021-01-16  91  78  65  27
    2021-01-17  32  27  88  51
    2021-01-18  82  61  24  30
    2021-01-19  38  31  61  17
    


```python
data = {'subject' : ['math', 'comp', 'phys', 'chem'],
       'score': [100, 90, 85, 95],
       'students': [94, 32, 83, 17]}
df = pd.DataFrame(data, columns = ['subject', 'score', 'students', 'class'],
                index = ['one', 'two', 'three', 'four'])

print('인덱스 직접부여:=>\n', df)
```

    인덱스 직접부여:=>
           subject  score  students class
    one      math    100        94   NaN
    two      comp     90        32   NaN
    three    phys     85        83   NaN
    four     chem     95        17   NaN
    


```python
print('컬럼리스트:df.columns=>', df.columns)
print('인덱스리스트:df.index=>', df.index)
print('keys:df.keys()=>', df.keys())
print('총항목개수:df.size=>', df.size)
print('차원수:df.ndim=>',df.ndim)
print('항목개수:df.shape=>',df.shape)
print('컬럼별항목개수:df.count()=>\n', df.count())
```

    컬럼리스트:df.columns=> Index(['subject', 'score', 'students', 'class'], dtype='object')
    인덱스리스트:df.index=> Index(['one', 'two', 'three', 'four'], dtype='object')
    keys:df.keys()=> Index(['subject', 'score', 'students', 'class'], dtype='object')
    총항목개수:df.size=> 16
    차원수:df.ndim=> 2
    항목개수:df.shape=> (4, 4)
    컬럼별항목개수:df.count()=>
     subject     4
    score       4
    students    4
    class       0
    dtype: int64
    


```python
print('columns 를 통해 값 가져옴 :df["score"] =>\n', df['score'])     # columns 값으로
print('\n columns 를 통해 값 가져옴 :df.score =>\n', df.score)          # columns 값으로 
print('\n index 를 통해 값 가져옴 :df.loc["one"] =>\n', df.loc['one']) # index 값으로
print('\n index 위치를 통해 가져옴 3번째 :df.iloc[2] =>\n', df.iloc[2])     # 인덱스 위치를 통해 엑세스
print('\n index 위치를 통해 가져옴 1~2번째 :df[1:3] =>\n', df[1:3])      # 인덱스 위치를 통해 
print("\n 위치를 통해 가져옴 1~2줄, 1~2컬럼 : => df.iloc[0:2, 0:2]\n",df.iloc[0:2, 0:2]) 
```

    columns 를 통해 값 가져옴 :df["score"] =>
     one      100
    two       90
    three     85
    four      95
    Name: score, dtype: int64
    
     columns 를 통해 값 가져옴 :df.score =>
     one      100
    two       90
    three     85
    four      95
    Name: score, dtype: int64
    
     index 를 통해 값 가져옴 :df.loc["one"] =>
     subject     math
    score        100
    students      94
    class        NaN
    Name: one, dtype: object
    
     index 위치를 통해 가져옴 3번째 :df.iloc[2] =>
     subject     phys
    score         85
    students      83
    class        NaN
    Name: three, dtype: object
    
     index 위치를 통해 가져옴 1~2번째 :df[1:3] =>
           subject  score  students class
    two      comp     90        32   NaN
    three    phys     85        83   NaN
    
     위치를 통해 가져옴 1~2줄, 1~2컬럼 : => df.iloc[0:2, 0:2]
         subject  score
    one    math    100
    two    comp     90
    


```python
print('score 가 90 초과인지?: df.score > 90 =>\n', df.score > 90)
print('\n score 가 90 초과인 것만: df[df.score > 90] =>\n', df[df.score > 90])

print("\n score 가 90 초과인 것의 2개 컬럼: df.loc[df['score'] > 90, ['subject', 'students']] =>\n", df.loc[df['score'] > 90, ['subject', 'students']]) 
```

    score 가 90 초과인지?: df.score > 90 =>
     one       True
    two      False
    three    False
    four      True
    Name: score, dtype: bool
    
     score 가 90 초과인 것만: df[df.score > 90] =>
          subject  score  students class
    one     math    100        94   NaN
    four    chem     95        17   NaN
    
     score 가 90 초과인 것의 2개 컬럼: df.loc[df['score'] > 90, ['subject', 'students']] =>
          subject  students
    one     math        94
    four    chem        17
    


```python
print('index 이름 : df.loc["one"] =>\n', df.loc['one'])   # loc[index 이름] 
print('\nindex 이름, columns 이름: df.loc["two", "subject"]  =>\n', df.loc['two', 'subject'])   # loc[index 이름, columns 이름]
print("\nindex 이름, columns 범위: df.loc['three', 'subject':'students'] =>\n", df.loc['three', 'subject':'students'])   # loc[index 이름, columns 범위]
```

    index 이름 : df.loc["one"] =>
     subject     math
    score        100
    students      94
    class        NaN
    Name: one, dtype: object
    
    index 이름, columns 이름: df.loc["two", "subject"]  =>
     comp
    
    index 이름, columns 범위: df.loc['three', 'subject':'students'] =>
     subject     phys
    score         85
    students      83
    Name: three, dtype: object
    


```python
print(df.index)
print(df.iloc[1].name)
print(df.iloc[1].values)
```

    Index(['one', 'two', 'three', 'four'], dtype='object')
    two
    ['comp' 90 32 nan]
    


```python
# 컬럼명 재부여
data = {'subject' : ['math', 'comp', 'phys', 'chem'],
       'score': [100, 95, 80, 90],
       'students': [87, 39, 50, 72]}

df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data, columns = ['subject', 'score', 'students', 'class'])
df3 = df2.copy()
df3.columns = ['subject1', 'score1', 'students1', 'class1']
print(df1)
print(df2)
print(df3)
```

      subject  score  students
    0    math    100        87
    1    comp     95        39
    2    phys     80        50
    3    chem     90        72
      subject  score  students class
    0    math    100        87   NaN
    1    comp     95        39   NaN
    2    phys     80        50   NaN
    3    chem     90        72   NaN
      subject1  score1  students1 class1
    0     math     100         87    NaN
    1     comp      95         39    NaN
    2     phys      80         50    NaN
    3     chem      90         72    NaN
    


```python
data2 = pd.DataFrame(data)
data2.loc[data2.shape[0]] = ['biol', 70, 33]  # 끝 줄에 데이터 추가
data2.loc['fifth', :] = ['lang', 75, 45] # 끝 줄에 데이터 추가
data2 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>math</td>
      <td>100.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comp</td>
      <td>95.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phys</td>
      <td>80.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chem</td>
      <td>90.0</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>biol</td>
      <td>70.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>fifth</th>
      <td>lang</td>
      <td>75.0</td>
      <td>45.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame(data, columns = ['subject', 'score', 'students', 'class'],
                    index=['a', 'b', 'c', 'd'])
df2 = df.copy()
print("컬럼삭제전 :                           =>", df2.columns)

del df2['students']
print("컬럼삭제후 : del df2['students']       =>", df2.columns)

df3 = df.copy()
df4 = df3.drop('score', axis=1)
print("컬럼삭제후 : df3.drop('score', axis=1) =>", df4.columns)
```

    컬럼삭제전 :                           => Index(['subject', 'score', 'students', 'class'], dtype='object')
    컬럼삭제후 : del df2['students']       => Index(['subject', 'score', 'class'], dtype='object')
    컬럼삭제후 : df3.drop('score', axis=1) => Index(['subject', 'students', 'class'], dtype='object')
    


```python
data = {'subject' : ['math', 'comp', 'phys', 'chem'],
       'score': [100, 95, 80, 90],
       'students': [87, 39, 50, 72]}
df = pd.DataFrame(data, columns = ['subject', 'score', 'students', 'class'],
                    index=['a', 'b', 'c', 'd'])
df2 = df.copy()
df2['class'] = [1, 2, 3, 4]
print(df)
print(df2)
```

      subject  score  students class
    a    math    100        87   NaN
    b    comp     95        39   NaN
    c    phys     80        50   NaN
    d    chem     90        72   NaN
      subject  score  students  class
    a    math    100        87      1
    b    comp     95        39      2
    c    phys     80        50      3
    d    chem     90        72      4
    


```python
# 인덱크 컬럼 바꿔줌
df1 = df.set_index('class')
print(df)
print(df1)
```

      subject  score  students class
    a    math    100        87   NaN
    b    comp     95        39   NaN
    c    phys     80        50   NaN
    d    chem     90        72   NaN
          subject  score  students
    class                         
    NaN      math    100        87
    NaN      comp     95        39
    NaN      phys     80        50
    NaN      chem     90        72
    


```python
# 인덱스 초기화
df1.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>math</td>
      <td>100</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>comp</td>
      <td>95</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>phys</td>
      <td>80</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>chem</td>
      <td>90</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Series 컬럼으로 추가
df5 = df.copy()
ser2 = pd.Series([5, 6, 7], index=['a', 'b', 'd'])
df5['class'] = ser2
df5 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>math</td>
      <td>100</td>
      <td>87</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>comp</td>
      <td>95</td>
      <td>39</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>phys</td>
      <td>80</td>
      <td>50</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>chem</td>
      <td>90</td>
      <td>72</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 정렬
df5 = df5.sort_values("score")
print(df5)
df5 = df5.sort_index() # 인덱스 정렬
print(df5)
```

      subject  score  students  class
    c    phys     80        50    NaN
    d    chem     90        72    7.0
    b    comp     95        39    6.0
    a    math    100        87    5.0
      subject  score  students  class
    a    math    100        87    5.0
    b    comp     95        39    6.0
    c    phys     80        50    NaN
    d    chem     90        72    7.0
    


```python
df6 = df.copy()
ser3 = pd.Series([9, 10, 11, 12], index = df6.index)
df6['class'] = ser3
df6
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>math</td>
      <td>100</td>
      <td>87</td>
      <td>9</td>
    </tr>
    <tr>
      <th>b</th>
      <td>comp</td>
      <td>95</td>
      <td>39</td>
      <td>10</td>
    </tr>
    <tr>
      <th>c</th>
      <td>phys</td>
      <td>80</td>
      <td>50</td>
      <td>11</td>
    </tr>
    <tr>
      <th>d</th>
      <td>chem</td>
      <td>90</td>
      <td>72</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df7 = df.copy()
df7['pass'] = df7.score >= 90
df7
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
      <th>class</th>
      <th>pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>math</td>
      <td>100</td>
      <td>87</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>b</th>
      <td>comp</td>
      <td>95</td>
      <td>39</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>c</th>
      <td>phys</td>
      <td>80</td>
      <td>50</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>d</th>
      <td>chem</td>
      <td>90</td>
      <td>72</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df8 = df.copy()
df8['average'] = [96.3, np.nan, 76.2, 88.6]
df8
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
      <th>class</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>math</td>
      <td>100</td>
      <td>87</td>
      <td>NaN</td>
      <td>96.3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>comp</td>
      <td>95</td>
      <td>39</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>phys</td>
      <td>80</td>
      <td>50</td>
      <td>NaN</td>
      <td>76.2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>chem</td>
      <td>90</td>
      <td>72</td>
      <td>NaN</td>
      <td>88.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# np.nan 값 삭제
df8.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
      <th>class</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# np.nan 값 다른 값으로 
df8.fillna(value=100.0) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>score</th>
      <th>students</th>
      <th>class</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>math</td>
      <td>100</td>
      <td>87</td>
      <td>100.0</td>
      <td>96.3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>comp</td>
      <td>95</td>
      <td>39</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>phys</td>
      <td>80</td>
      <td>50</td>
      <td>100.0</td>
      <td>76.2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>chem</td>
      <td>90</td>
      <td>72</td>
      <td>100.0</td>
      <td>88.6</td>
    </tr>
  </tbody>
</table>
</div>



## 연산


```python
a = pd.Series([1, 2, 3, 4])
print(a)
print(a+1)
print(a-1)
print(a*2)
print(a/2)
print(a//2)
print(a%2)
print(a**2)
print(2**a) 
```

    0    1
    1    2
    2    3
    3    4
    dtype: int64
    0    2
    1    3
    2    4
    3    5
    dtype: int64
    0    0
    1    1
    2    2
    3    3
    dtype: int64
    0    2
    1    4
    2    6
    3    8
    dtype: int64
    0    0.5
    1    1.0
    2    1.5
    3    2.0
    dtype: float64
    0    0
    1    1
    2    1
    3    2
    dtype: int64
    0    1
    1    0
    2    1
    3    0
    dtype: int64
    0     1
    1     4
    2     9
    3    16
    dtype: int64
    0     2
    1     4
    2     8
    3    16
    dtype: int64
    


```python
a = pd.Series([2, 3, 4, 5], index=['d', 'b', 'a', 'c'])
b = pd.Series([1, 2, 3, 4])
print(a+b)
a = pd.Series([2, 3, 4, 5], index=['d', 'b', 'a', 'c'])
b = pd.Series([1, 2, 3, 4], index=a.index)
print(a+b)
```

    0   NaN
    1   NaN
    2   NaN
    3   NaN
    a   NaN
    b   NaN
    c   NaN
    d   NaN
    dtype: float64
    d    3
    b    5
    a    7
    c    9
    dtype: int64
    


```python
import pandas as pd
data = {'math': 100, 'comp': 95, 'phys': 85, 'chem': 90}
subj = ['math', 'biol', 'phys', 'chem']

c = pd.Series(data)
d = pd.Series(data, index=subj)

print('c =>',c,'d =>',d,'c+d =>',c+d,'c*d =>',c*d,sep="\n")
```

    c =>
    math    100
    comp     95
    phys     85
    chem     90
    dtype: int64
    d =>
    math    100.0
    biol      NaN
    phys     85.0
    chem     90.0
    dtype: float64
    c+d =>
    biol      NaN
    chem    180.0
    comp      NaN
    math    200.0
    phys    170.0
    dtype: float64
    c*d =>
    biol        NaN
    chem     8100.0
    comp        NaN
    math    10000.0
    phys     7225.0
    dtype: float64
    


```python
x = pd.DataFrame(np.arange(12).reshape(3, 4), columns = list('abcd'))
print(x)
print(x+1)
print(x-1)
print(x*2)
print(x/2)
print(x//2)
print(x%2)
print(x**2)
print(2**x) 
```

       a  b   c   d
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
       a   b   c   d
    0  1   2   3   4
    1  5   6   7   8
    2  9  10  11  12
       a  b  c   d
    0 -1  0  1   2
    1  3  4  5   6
    2  7  8  9  10
        a   b   c   d
    0   0   2   4   6
    1   8  10  12  14
    2  16  18  20  22
         a    b    c    d
    0  0.0  0.5  1.0  1.5
    1  2.0  2.5  3.0  3.5
    2  4.0  4.5  5.0  5.5
       a  b  c  d
    0  0  0  1  1
    1  2  2  3  3
    2  4  4  5  5
       a  b  c  d
    0  0  1  0  1
    1  0  1  0  1
    2  0  1  0  1
        a   b    c    d
    0   0   1    4    9
    1  16  25   36   49
    2  64  81  100  121
         a    b     c     d
    0    1    2     4     8
    1   16   32    64   128
    2  256  512  1024  2048
    


```python
np.arange(12).reshape(3, 4)
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
np.array(list(range(12))).reshape(4,3)
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])




```python
x = pd.DataFrame(np.arange(12).reshape(3, 4), columns = list('abcd'))
y = pd.DataFrame(np.arange(20).reshape(4, 5), columns = list('abcde'))
print(y)
print(x+y)
print(x-y)
print(x*y)
```

        a   b   c   d   e
    0   0   1   2   3   4
    1   5   6   7   8   9
    2  10  11  12  13  14
    3  15  16  17  18  19
          a     b     c     d   e
    0   0.0   2.0   4.0   6.0 NaN
    1   9.0  11.0  13.0  15.0 NaN
    2  18.0  20.0  22.0  24.0 NaN
    3   NaN   NaN   NaN   NaN NaN
         a    b    c    d   e
    0  0.0  0.0  0.0  0.0 NaN
    1 -1.0 -1.0 -1.0 -1.0 NaN
    2 -2.0 -2.0 -2.0 -2.0 NaN
    3  NaN  NaN  NaN  NaN NaN
          a     b      c      d   e
    0   0.0   1.0    4.0    9.0 NaN
    1  20.0  30.0   42.0   56.0 NaN
    2  80.0  99.0  120.0  143.0 NaN
    3   NaN   NaN    NaN    NaN NaN
    


```python
z = pd.DataFrame(np.arange(12).reshape(4, 3),
                columns = list('abc'),
                index= ['math', 'comp', 'phys', 'chem'])
e = z.iloc[0]

print(z)
print(e)
print(z+e)
print(z-e) 
```

          a   b   c
    math  0   1   2
    comp  3   4   5
    phys  6   7   8
    chem  9  10  11
    a    0
    b    1
    c    2
    Name: math, dtype: int32
          a   b   c
    math  0   2   4
    comp  3   5   7
    phys  6   8  10
    chem  9  11  13
          a  b  c
    math  0  0  0
    comp  3  3  3
    phys  6  6  6
    chem  9  9  9
    


```python
f = pd.Series(range(3), index = list('bde'))
print(z)
print(f)
print(z+f)
```

          a   b   c
    math  0   1   2
    comp  3   4   5
    phys  6   7   8
    chem  9  10  11
    b    0
    d    1
    e    2
    dtype: int64
           a     b   c   d   e
    math NaN   1.0 NaN NaN NaN
    comp NaN   4.0 NaN NaN NaN
    phys NaN   7.0 NaN NaN NaN
    chem NaN  10.0 NaN NaN NaN
    


```python
print(x)
print(y)
print(x.add(y))
```

       a  b   c   d
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
        a   b   c   d   e
    0   0   1   2   3   4
    1   5   6   7   8   9
    2  10  11  12  13  14
    3  15  16  17  18  19
          a     b     c     d   e
    0   0.0   2.0   4.0   6.0 NaN
    1   9.0  11.0  13.0  15.0 NaN
    2  18.0  20.0  22.0  24.0 NaN
    3   NaN   NaN   NaN   NaN NaN
    


```python
print(x.add(y, fill_value=100))
```

           a      b      c      d      e
    0    0.0    2.0    4.0    6.0  104.0
    1    9.0   11.0   13.0   15.0  109.0
    2   18.0   20.0   22.0   24.0  114.0
    3  115.0  116.0  117.0  118.0  119.0
    


```python
print(y.add(x, fill_value=100))
```

           a      b      c      d      e
    0    0.0    2.0    4.0    6.0  104.0
    1    9.0   11.0   13.0   15.0  109.0
    2   18.0   20.0   22.0   24.0  114.0
    3  115.0  116.0  117.0  118.0  119.0
    


```python
print(x.sub(y))
print(x.mul(y))
print(x.div(y))
print(x.pow(y))
print(x.mod(y)) 
```

         a    b    c    d   e
    0  0.0  0.0  0.0  0.0 NaN
    1 -1.0 -1.0 -1.0 -1.0 NaN
    2 -2.0 -2.0 -2.0 -2.0 NaN
    3  NaN  NaN  NaN  NaN NaN
          a     b      c      d   e
    0   0.0   1.0    4.0    9.0 NaN
    1  20.0  30.0   42.0   56.0 NaN
    2  80.0  99.0  120.0  143.0 NaN
    3   NaN   NaN    NaN    NaN NaN
         a         b         c         d   e
    0  NaN  1.000000  1.000000  1.000000 NaN
    1  0.8  0.833333  0.857143  0.875000 NaN
    2  0.8  0.818182  0.833333  0.846154 NaN
    3  NaN       NaN       NaN       NaN NaN
                  a             b             c             d   e
    0  1.000000e+00  1.000000e+00  4.000000e+00  2.700000e+01 NaN
    1  1.024000e+03  1.562500e+04  2.799360e+05  5.764801e+06 NaN
    2  1.073742e+09  3.138106e+10  1.000000e+12  3.452271e+13 NaN
    3           NaN           NaN           NaN           NaN NaN
         a    b     c     d   e
    0  NaN  0.0   0.0   0.0 NaN
    1  4.0  5.0   6.0   7.0 NaN
    2  8.0  9.0  10.0  11.0 NaN
    3  NaN  NaN   NaN   NaN NaN
    


```python
print(x)
print(x.diff())
print(x.diff(axis=0))
print(x.diff(axis=1))
```

       a  b   c   d
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
         a    b    c    d
    0  NaN  NaN  NaN  NaN
    1  4.0  4.0  4.0  4.0
    2  4.0  4.0  4.0  4.0
         a    b    c    d
    0  NaN  NaN  NaN  NaN
    1  4.0  4.0  4.0  4.0
    2  4.0  4.0  4.0  4.0
        a    b    c    d
    0 NaN  1.0  1.0  1.0
    1 NaN  1.0  1.0  1.0
    2 NaN  1.0  1.0  1.0
    


```python
x = x*(-1)
print(x)
print(x.abs()) 
```

       a  b   c   d
    0  0 -1  -2  -3
    1 -4 -5  -6  -7
    2 -8 -9 -10 -11
       a  b   c   d
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
    

# pandas 통계함수


```python
import numpy as np
data = [[1.4, np.nan],
       [8.3, -2.1],
       [0.02, -1.11],
       [np.nan, np.nan]]
x = pd.DataFrame(data, columns=['one', 'two'], index=['a', 'b', 'c', 'd'])
x 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>8.30</td>
      <td>-2.10</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.02</td>
      <td>-1.11</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(x.sum())
print(x.sum(axis=0))
print(x.sum(axis=1))
```

    one    9.72
    two   -3.21
    dtype: float64
    one    9.72
    two   -3.21
    dtype: float64
    a    1.40
    b    6.20
    c   -1.09
    d    0.00
    dtype: float64
    


```python
print(x.sum(axis=0, skipna=False))
print(x.sum(axis=1, skipna=False))
```

    one   NaN
    two   NaN
    dtype: float64
    a     NaN
    b    6.20
    c   -1.09
    d     NaN
    dtype: float64
    


```python
print(x['one'].sum())
print(x.loc['c'].sum())
```

    9.72
    -1.09
    


```python
print(x)
print('count 함수는 NaN을 제외한 전체 개수 =>\n', x.count())
print("mean 함수는 평균 =>\n", x.mean())
print("median 함수는 중간값 =>\n", x.median())
print("mad 함수는 absolute deviation의 평균 =>\n", x.mad())    
```

        one   two
    a  1.40   NaN
    b  8.30 -2.10
    c  0.02 -1.11
    d   NaN   NaN
    count 함수는 NaN을 제외한 전체 개수 =>
     one    3
    two    2
    dtype: int64
    mean 함수는 평균 =>
     one    3.240
    two   -1.605
    dtype: float64
    median 함수는 중간값 =>
     one    1.400
    two   -1.605
    dtype: float64
    mad 함수는 absolute deviation의 평균 =>
     one    3.373333
    two    0.495000
    dtype: float64
    


```python
y = pd.DataFrame(np.random.randint(0, 100, (6, 4)),
                columns = list('ABCD'),
                index = pd.date_range('20200114', periods=6))
print(y)
print('상관계수 : corr => \n', y.corr())
print("상관계수 : y['A'].corr(y['B']):=>", y['A'].corr(y['B']))
print("공분산 : cov => \n", y.cov())
print("공분산 : y['A'].cov(y['C']) => ", y['A'].cov(y['C'])) 
```

                 A   B   C   D
    2020-01-14  77   6  28  41
    2020-01-15  45  63  80  25
    2020-01-16  52  97   0   9
    2020-01-17  21   7  65  71
    2020-01-18  80   5  45  45
    2020-01-19  22  85  82   4
    상관계수 : corr => 
               A         B         C         D
    A  1.000000 -0.411749 -0.565127  0.086255
    B -0.411749  1.000000 -0.042424 -0.893428
    C -0.565127 -0.042424  1.000000  0.091068
    D  0.086255 -0.893428  0.091068  1.000000
    상관계수 : y['A'].corr(y['B']):=> -0.4117491080365865
    공분산 : cov => 
            A            B       C      D
    A  656.3  -452.100000  -465.0   55.3
    B -452.1  1836.966667   -58.4 -958.3
    C -465.0   -58.400000  1031.6   73.2
    D   55.3  -958.300000    73.2  626.3
    공분산 : y['A'].cov(y['C']) =>  -465.0
    


```python
print(y)
print("std는 표준편차 => \n", y.std())
print("var은 분산 =>\n", y.var())
print("min은 최솟값 =>\n", y.min())
print("max는 최댓값 =>\n", y.max())
print("idxmin은 최솟값의 인덱스 =>\n", y.idxmin())
print("idxmax는 최댓값의 인덱스 =>\n", y.idxmax())
```

                 A   B   C   D
    2020-01-14  77   6  28  41
    2020-01-15  45  63  80  25
    2020-01-16  52  97   0   9
    2020-01-17  21   7  65  71
    2020-01-18  80   5  45  45
    2020-01-19  22  85  82   4
    std는 표준편차 => 
     A    25.618353
    B    42.859849
    C    32.118530
    D    25.025986
    dtype: float64
    var은 분산 =>
     A     656.300000
    B    1836.966667
    C    1031.600000
    D     626.300000
    dtype: float64
    min은 최솟값 =>
     A    21
    B     5
    C     0
    D     4
    dtype: int32
    max는 최댓값 =>
     A    80
    B    97
    C    82
    D    71
    dtype: int32
    idxmin은 최솟값의 인덱스 =>
     A   2020-01-17
    B   2020-01-18
    C   2020-01-16
    D   2020-01-19
    dtype: datetime64[ns]
    idxmax는 최댓값의 인덱스 =>
     A   2020-01-18
    B   2020-01-16
    C   2020-01-19
    D   2020-01-17
    dtype: datetime64[ns]
    


```python
print("quantile은 사분위수 =>\n", y.quantile())
print("cumsum은 누적합 => \n", y.cumsum())
print("cumprod는 누적곱 => \n", y.cumprod())
```

    quantile은 사분위수 =>
     A    48.5
    B    35.0
    C    55.0
    D    33.0
    Name: 0.5, dtype: float64
    cumsum은 누적합 => 
                   A    B    C    D
    2020-01-14   77    6   28   41
    2020-01-15  122   69  108   66
    2020-01-16  174  166  108   75
    2020-01-17  195  173  173  146
    2020-01-18  275  178  218  191
    2020-01-19  297  263  300  195
    cumprod는 누적곱 => 
                          A          B     C          D
    2020-01-14          77          6    28         41
    2020-01-15        3465        378  2240       1025
    2020-01-16      180180      36666     0       9225
    2020-01-17     3783780     256662     0     654975
    2020-01-18   302702400    1283310     0   29473875
    2020-01-19 -1930481792  109081350     0  117895500
    

# pandas의 논리 함수와 기타 유용한 함수


```python
print(x)
print(pd.isnull(x))
print(x.isnull())
```

        one   two
    a  1.40   NaN
    b  8.30 -2.10
    c  0.02 -1.11
    d   NaN   NaN
         one    two
    a  False   True
    b  False  False
    c  False  False
    d   True   True
         one    two
    a  False   True
    b  False  False
    c  False  False
    d   True   True
    


```python
print(x['one'].isnull())
print(x.isnull().sum()) 
print(x['one'].isnull().sum())
```

    a    False
    b    False
    c    False
    d     True
    Name: one, dtype: bool
    one    1
    two    2
    dtype: int64
    1
    


```python
print(x)
print(pd.notnull(x))
print(x.notnull())
```

        one   two
    a  1.40   NaN
    b  8.30 -2.10
    c  0.02 -1.11
    d   NaN   NaN
         one    two
    a   True  False
    b   True   True
    c   True   True
    d  False  False
         one    two
    a   True  False
    b   True   True
    c   True   True
    d  False  False
    


```python
print(x)
print(x.isnull().any())
print(x.isnull().all()) 
```

        one   two
    a  1.40   NaN
    b  8.30 -2.10
    c  0.02 -1.11
    d   NaN   NaN
    one    True
    two    True
    dtype: bool
    one    False
    two    False
    dtype: bool
    


```python
s1 = pd.date_range('20210114', periods=6)
s1 = np.random.permutation(s1)
s2 = list('DBCA')
y = pd.DataFrame(np.random.randint(0, 100, (6, 4)), index=s1, columns=s2)
y 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>B</th>
      <th>C</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-17</th>
      <td>32</td>
      <td>47</td>
      <td>55</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2021-01-18</th>
      <td>6</td>
      <td>70</td>
      <td>73</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2021-01-14</th>
      <td>21</td>
      <td>27</td>
      <td>98</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2021-01-19</th>
      <td>33</td>
      <td>86</td>
      <td>8</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2021-01-16</th>
      <td>57</td>
      <td>74</td>
      <td>27</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2021-01-15</th>
      <td>14</td>
      <td>15</td>
      <td>92</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.sort_index(axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>B</th>
      <th>C</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-14</th>
      <td>21</td>
      <td>27</td>
      <td>98</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2021-01-15</th>
      <td>14</td>
      <td>15</td>
      <td>92</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2021-01-16</th>
      <td>57</td>
      <td>74</td>
      <td>27</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2021-01-17</th>
      <td>32</td>
      <td>47</td>
      <td>55</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2021-01-18</th>
      <td>6</td>
      <td>70</td>
      <td>73</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2021-01-19</th>
      <td>33</td>
      <td>86</td>
      <td>8</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.sort_values(by='D')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>B</th>
      <th>C</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-18</th>
      <td>6</td>
      <td>70</td>
      <td>73</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2021-01-15</th>
      <td>14</td>
      <td>15</td>
      <td>92</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2021-01-14</th>
      <td>21</td>
      <td>27</td>
      <td>98</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2021-01-17</th>
      <td>32</td>
      <td>47</td>
      <td>55</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2021-01-19</th>
      <td>33</td>
      <td>86</td>
      <td>8</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2021-01-16</th>
      <td>57</td>
      <td>74</td>
      <td>27</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.sort_values(by='C', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>B</th>
      <th>C</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-14</th>
      <td>21</td>
      <td>27</td>
      <td>98</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2021-01-15</th>
      <td>14</td>
      <td>15</td>
      <td>92</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2021-01-18</th>
      <td>6</td>
      <td>70</td>
      <td>73</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2021-01-17</th>
      <td>32</td>
      <td>47</td>
      <td>55</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2021-01-16</th>
      <td>57</td>
      <td>74</td>
      <td>27</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2021-01-19</th>
      <td>33</td>
      <td>86</td>
      <td>8</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
</div>




```python
y['E'] = np.random.randint(0, 3, size=6)
y
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>B</th>
      <th>C</th>
      <th>A</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-17</th>
      <td>32</td>
      <td>47</td>
      <td>55</td>
      <td>76</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2021-01-18</th>
      <td>6</td>
      <td>70</td>
      <td>73</td>
      <td>68</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2021-01-14</th>
      <td>21</td>
      <td>27</td>
      <td>98</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2021-01-19</th>
      <td>33</td>
      <td>86</td>
      <td>8</td>
      <td>89</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2021-01-16</th>
      <td>57</td>
      <td>74</td>
      <td>27</td>
      <td>84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2021-01-15</th>
      <td>14</td>
      <td>15</td>
      <td>92</td>
      <td>88</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y['E'].unique()
```




    array([2, 1, 0])




```python
y.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2021-01-17</th>
      <th>2021-01-18</th>
      <th>2021-01-14</th>
      <th>2021-01-19</th>
      <th>2021-01-16</th>
      <th>2021-01-15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>D</th>
      <td>32</td>
      <td>6</td>
      <td>21</td>
      <td>33</td>
      <td>57</td>
      <td>14</td>
    </tr>
    <tr>
      <th>B</th>
      <td>47</td>
      <td>70</td>
      <td>27</td>
      <td>86</td>
      <td>74</td>
      <td>15</td>
    </tr>
    <tr>
      <th>C</th>
      <td>55</td>
      <td>73</td>
      <td>98</td>
      <td>8</td>
      <td>27</td>
      <td>92</td>
    </tr>
    <tr>
      <th>A</th>
      <td>76</td>
      <td>68</td>
      <td>62</td>
      <td>89</td>
      <td>84</td>
      <td>88</td>
    </tr>
    <tr>
      <th>E</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y['B'].isin([91,1])
```




    2021-01-17    False
    2021-01-18    False
    2021-01-14    False
    2021-01-19    False
    2021-01-16    False
    2021-01-15    False
    Name: B, dtype: bool



# DataFrame 결합


```python
x1 = pd.DataFrame(np.random.randint(0, 100, (3, 4)),
                 columns = list('ABCD'),
                 index = np.arange(3))
x2 = pd.DataFrame(np.random.randint(100, 200, (3, 4)),
                 columns = list('ABCD'),
                 index = np.arange(3)+3)
print(x1)
print(x2) 
```

        A   B   C   D
    0  41  35   6  29
    1  18  50  36  68
    2  61  41  57   4
         A    B    C    D
    3  128  158  194  153
    4  161  141  103  103
    5  180  122  142  109
    


```python
pd.concat([x1, x2])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>35</td>
      <td>6</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>50</td>
      <td>36</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61</td>
      <td>41</td>
      <td>57</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>128</td>
      <td>158</td>
      <td>194</td>
      <td>153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161</td>
      <td>141</td>
      <td>103</td>
      <td>103</td>
    </tr>
    <tr>
      <th>5</th>
      <td>180</td>
      <td>122</td>
      <td>142</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([x1, x2], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41.0</td>
      <td>35.0</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.0</td>
      <td>50.0</td>
      <td>36.0</td>
      <td>68.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61.0</td>
      <td>41.0</td>
      <td>57.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>128.0</td>
      <td>158.0</td>
      <td>194.0</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
      <td>141.0</td>
      <td>103.0</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>180.0</td>
      <td>122.0</td>
      <td>142.0</td>
      <td>109.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
x3 = pd.DataFrame(np.random.randint(200, 300, (3, 4)),
                 columns = list('EFGH'),
                 index = np.arange(3))
x4 = pd.DataFrame(np.random.randint(300, 400, (3, 4)),
                 columns = list('ABCE'),
                 index = [0, 1, 3])
print(x3)
print(x4) 
```

         E    F    G    H
    0  230  283  233  218
    1  215  228  232  238
    2  267  295  264  297
         A    B    C    E
    0  379  304  352  378
    1  371  371  322  341
    3  348  379  304  358
    


```python
print(x1)
print(x4)
pd.concat([x1, x4], join='outer')
```

        A   B   C   D
    0  41  35   6  29
    1  18  50  36  68
    2  61  41  57   4
         A    B    C    E
    0  379  304  352  378
    1  371  371  322  341
    3  348  379  304  358
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>35</td>
      <td>6</td>
      <td>29.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>50</td>
      <td>36</td>
      <td>68.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61</td>
      <td>41</td>
      <td>57</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>379</td>
      <td>304</td>
      <td>352</td>
      <td>NaN</td>
      <td>378.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>371</td>
      <td>371</td>
      <td>322</td>
      <td>NaN</td>
      <td>341.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>348</td>
      <td>379</td>
      <td>304</td>
      <td>NaN</td>
      <td>358.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([x1, x4], join='inner')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>50</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61</td>
      <td>41</td>
      <td>57</td>
    </tr>
    <tr>
      <th>0</th>
      <td>379</td>
      <td>304</td>
      <td>352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>371</td>
      <td>371</td>
      <td>322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>348</td>
      <td>379</td>
      <td>304</td>
    </tr>
  </tbody>
</table>
</div>




```python
x5 = pd.DataFrame(np.random.randint(0, 100, (3, 4)),
                 columns = list('ABCD'),
                 index = list('abc'))
x6 = pd.DataFrame(np.random.randint(100, 200, (3, 4)),
                 columns = list('ABCD'),
                 index = list('def'))
print(x5)
print(x6) 
```

        A   B   C   D
    a  65  39  47  44
    b  83  39  60  10
    c  38  97  90  17
         A    B    C    D
    d  191  167  176  111
    e  168  171  187  123
    f  138  186  137  146
    


```python
pd.concat([x5, x6], ignore_index=False) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>65</td>
      <td>39</td>
      <td>47</td>
      <td>44</td>
    </tr>
    <tr>
      <th>b</th>
      <td>83</td>
      <td>39</td>
      <td>60</td>
      <td>10</td>
    </tr>
    <tr>
      <th>c</th>
      <td>38</td>
      <td>97</td>
      <td>90</td>
      <td>17</td>
    </tr>
    <tr>
      <th>d</th>
      <td>191</td>
      <td>167</td>
      <td>176</td>
      <td>111</td>
    </tr>
    <tr>
      <th>e</th>
      <td>168</td>
      <td>171</td>
      <td>187</td>
      <td>123</td>
    </tr>
    <tr>
      <th>f</th>
      <td>138</td>
      <td>186</td>
      <td>137</td>
      <td>146</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([x5, x6], ignore_index=True)  # 인덱스가 새로 부여
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65</td>
      <td>39</td>
      <td>47</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83</td>
      <td>39</td>
      <td>60</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>97</td>
      <td>90</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>191</td>
      <td>167</td>
      <td>176</td>
      <td>111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>168</td>
      <td>171</td>
      <td>187</td>
      <td>123</td>
    </tr>
    <tr>
      <th>5</th>
      <td>138</td>
      <td>186</td>
      <td>137</td>
      <td>146</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([x5, x6], keys=['x5', 'x6'])  # 인덱스에 계층을 부여하고 싶으면 keys 속성을 넣습니다 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">x5</th>
      <th>a</th>
      <td>65</td>
      <td>39</td>
      <td>47</td>
      <td>44</td>
    </tr>
    <tr>
      <th>b</th>
      <td>83</td>
      <td>39</td>
      <td>60</td>
      <td>10</td>
    </tr>
    <tr>
      <th>c</th>
      <td>38</td>
      <td>97</td>
      <td>90</td>
      <td>17</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">x6</th>
      <th>d</th>
      <td>191</td>
      <td>167</td>
      <td>176</td>
      <td>111</td>
    </tr>
    <tr>
      <th>e</th>
      <td>168</td>
      <td>171</td>
      <td>187</td>
      <td>123</td>
    </tr>
    <tr>
      <th>f</th>
      <td>138</td>
      <td>186</td>
      <td>137</td>
      <td>146</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([x5, x6], keys=['x5', 'x6'],
         names=['name', 'num']) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
    <tr>
      <th>name</th>
      <th>num</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">x5</th>
      <th>a</th>
      <td>65</td>
      <td>39</td>
      <td>47</td>
      <td>44</td>
    </tr>
    <tr>
      <th>b</th>
      <td>83</td>
      <td>39</td>
      <td>60</td>
      <td>10</td>
    </tr>
    <tr>
      <th>c</th>
      <td>38</td>
      <td>97</td>
      <td>90</td>
      <td>17</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">x6</th>
      <th>d</th>
      <td>191</td>
      <td>167</td>
      <td>176</td>
      <td>111</td>
    </tr>
    <tr>
      <th>e</th>
      <td>168</td>
      <td>171</td>
      <td>187</td>
      <td>123</td>
    </tr>
    <tr>
      <th>f</th>
      <td>138</td>
      <td>186</td>
      <td>137</td>
      <td>146</td>
    </tr>
  </tbody>
</table>
</div>




```python
x7 = pd.DataFrame(np.random.randint(0, 100, (3, 4)),
                 columns = list('ABCD'),
                 index = list('abc'))
x8 = pd.DataFrame(np.random.randint(100, 200, (3, 4)),
                 columns = list('ABCD'),
                 index = list('cde'))
print(x7)
print(x8)
```

        A   B   C   D
    a  14  74  80  72
    b  81   4  69  86
    c  27  93  78   4
         A    B    C    D
    c  164  105  167  163
    d  143  180  109  132
    e  119  102  111  161
    


```python
pd.concat([x7, x8], verify_integrity=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>14</td>
      <td>74</td>
      <td>80</td>
      <td>72</td>
    </tr>
    <tr>
      <th>b</th>
      <td>81</td>
      <td>4</td>
      <td>69</td>
      <td>86</td>
    </tr>
    <tr>
      <th>c</th>
      <td>27</td>
      <td>93</td>
      <td>78</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>164</td>
      <td>105</td>
      <td>167</td>
      <td>163</td>
    </tr>
    <tr>
      <th>d</th>
      <td>143</td>
      <td>180</td>
      <td>109</td>
      <td>132</td>
    </tr>
    <tr>
      <th>e</th>
      <td>119</td>
      <td>102</td>
      <td>111</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pd.concat([x7, x8], verify_integrity=True)   # verify_integrity에서 인덱스의 중복 여부를 판단하여 오류를 발생
```

# Series 결합


```python
x1 = pd.Series([0, 1, 2])
x2 = pd.Series([3, 4, 5])
x3 = pd.Series([6, 7, 8])
pd.concat([x1, x2, x3]) 
```




    0    0
    1    1
    2    2
    0    3
    1    4
    2    5
    0    6
    1    7
    2    8
    dtype: int64




```python
x1.append(x2).append(x3)
```




    0    0
    1    1
    2    2
    0    3
    1    4
    2    5
    0    6
    1    7
    2    8
    dtype: int64




```python
pd.concat([x1, x2, x3], axis=1) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([x1, x2, x3], axis=1, keys=['a', 'b', 'c'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
s1 = pd.Series(np.arange(3), name='S1')
df1 = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns = list('ABCD'),
                  index = np.arange(3))
pd.concat([s1, df1], axis=1) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S1</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([s1, df1], axis=1, ignore_index=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
s2 = pd.Series(np.arange(4), index=list('ABCE'))
df1.append(s2, ignore_index=True) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



# merge 함수 결합


```python
df1 = pd.DataFrame({'a' : [0, 1, 2, 3],
                   'b' : [10, 11, 12, 13],
                   'c' : [20, 21, 22, 23]})
df2 = pd.DataFrame({'a' : [2, 3, 4, 5],
                   'c' : [32, 33, 34, 35],
                   'd' : [42, 43, 44, 45]})
print(df1)
print(df2) 
```

       a   b   c
    0  0  10  20
    1  1  11  21
    2  2  12  22
    3  3  13  23
       a   c   d
    0  2  32  42
    1  3  33  43
    2  4  34  44
    3  5  35  45
    


```python
print(pd.merge(df1, df2, how='left', on='a'))
print(pd.merge(df1, df2, how='right', on='a')) 
# merge 는 dataframe df1과 df2를 합치겠다는 뜻
# how='left', 'right'는 어떤 dataframe을 기준으로 합칠지를 결정
# on='a'는 기준이 되는 dataframe의 'a'에 있는 값만 가져오겠다는 뜻
# 첫번째 코드를 보면 how='left' 이고 on='a' 이기 때문에 df1의 a인 0, 1, 2, 3만 가져옴
# 반대로 두 번째 코드는 2, 3, 4, 5만 가져옴
```

       a   b  c_x   c_y     d
    0  0  10   20   NaN   NaN
    1  1  11   21   NaN   NaN
    2  2  12   22  32.0  42.0
    3  3  13   23  33.0  43.0
       a     b   c_x  c_y   d
    0  2  12.0  22.0   32  42
    1  3  13.0  23.0   33  43
    2  4   NaN   NaN   34  44
    3  5   NaN   NaN   35  45
    


```python
print(pd.merge(df1, df2, on='a'))
print(pd.merge(df1, df2, how='outer', on='a'))
print(pd.merge(df1, df2, how='inner', on='a')) 
```

       a   b  c_x  c_y   d
    0  2  12   22   32  42
    1  3  13   23   33  43
       a     b   c_x   c_y     d
    0  0  10.0  20.0   NaN   NaN
    1  1  11.0  21.0   NaN   NaN
    2  2  12.0  22.0  32.0  42.0
    3  3  13.0  23.0  33.0  43.0
    4  4   NaN   NaN  34.0  44.0
    5  5   NaN   NaN  35.0  45.0
       a   b  c_x  c_y   d
    0  2  12   22   32  42
    1  3  13   23   33  43
    


```python
pd.merge(df1, df2, how='outer', on='a', indicator=True) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c_x</th>
      <th>c_y</th>
      <th>d</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>13.0</td>
      <td>23.0</td>
      <td>33.0</td>
      <td>43.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>45.0</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1, df2, how='outer', on='a', indicator='info') 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c_x</th>
      <th>c_y</th>
      <th>d</th>
      <th>info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>13.0</td>
      <td>23.0</td>
      <td>33.0</td>
      <td>43.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>45.0</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1, df2, how='inner', on='a', suffixes=('left', 'right'))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>cleft</th>
      <th>cright</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>12</td>
      <td>22</td>
      <td>32</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>13</td>
      <td>23</td>
      <td>33</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



# join 함수


```python
df1 = pd.DataFrame({'a': [10, 11, 12, 13],
                   'b': [20, 21, 22, 23]},
                  index = [0, 1, 2, 3])
df2 = pd.DataFrame({'c': [32, 33, 34, 35],
                   'd': [42, 43, 44, 45]},
                  index = [2, 3, 4, 5])
print(df1)
print(df2) 
```

        a   b
    0  10  20
    1  11  21
    2  12  22
    3  13  23
        c   d
    2  32  42
    3  33  43
    4  34  44
    5  35  45
    


```python
print(pd.merge(df1, df2, left_index=True, right_index=True, how='left'))
print(df1.join(df2, how='left')) 
```

        a   b     c     d
    0  10  20   NaN   NaN
    1  11  21   NaN   NaN
    2  12  22  32.0  42.0
    3  13  23  33.0  43.0
        a   b     c     d
    0  10  20   NaN   NaN
    1  11  21   NaN   NaN
    2  12  22  32.0  42.0
    3  13  23  33.0  43.0
    


```python
print(pd.merge(df1, df2, left_index=True, right_index=True, how='right'))
print(df1.join(df2, how='right')) 
```

          a     b   c   d
    2  12.0  22.0  32  42
    3  13.0  23.0  33  43
    4   NaN   NaN  34  44
    5   NaN   NaN  35  45
          a     b   c   d
    2  12.0  22.0  32  42
    3  13.0  23.0  33  43
    4   NaN   NaN  34  44
    5   NaN   NaN  35  45
    


```python
print(pd.merge(df1, df2, left_index=True, right_index=True, how='inner'))
print(df1.join(df2, how='inner'))
print(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'))
print(df1.join(df2, how='outer')) 

# merge 함수와 join 함수의 차이를 보면, 
# merge 함수의 경우 on 속성을 이용하여 열을 기준으로 결합을 진행
# join 함수의 경우에는 left_index, right_index를 이용하여 인덱스를 기준으로 결합
```

        a   b   c   d
    2  12  22  32  42
    3  13  23  33  43
        a   b   c   d
    2  12  22  32  42
    3  13  23  33  43
          a     b     c     d
    0  10.0  20.0   NaN   NaN
    1  11.0  21.0   NaN   NaN
    2  12.0  22.0  32.0  42.0
    3  13.0  23.0  33.0  43.0
    4   NaN   NaN  34.0  44.0
    5   NaN   NaN  35.0  45.0
          a     b     c     d
    0  10.0  20.0   NaN   NaN
    1  11.0  21.0   NaN   NaN
    2  12.0  22.0  32.0  42.0
    3  13.0  23.0  33.0  43.0
    4   NaN   NaN  34.0  44.0
    5   NaN   NaN  35.0  45.0
    


```python
df3 = pd.DataFrame({'i': [0, 1, 2, 3],
                   'a': [10, 11, 12, 13],
                   'b': [20, 21, 22, 23]})
df4 = pd.DataFrame({'c': [32, 33, 34, 35],
                   'd': [42, 43, 44, 45]},
                  index = [2, 3, 4, 5])
print(df3)
print(df4) 
```

       i   a   b
    0  0  10  20
    1  1  11  21
    2  2  12  22
    3  3  13  23
        c   d
    2  32  42
    3  33  43
    4  34  44
    5  35  45
    


```python
print(pd.merge(df3, df4, left_on='i', right_index=True, how='left'))
print(df3.join(df4, on='i', how='left'))
print(pd.merge(df3, df4, left_on='i', right_index=True, how='right'))
print(df3.join(df4, on='i', how='right')) 
```

       i   a   b     c     d
    0  0  10  20   NaN   NaN
    1  1  11  21   NaN   NaN
    2  2  12  22  32.0  42.0
    3  3  13  23  33.0  43.0
       i   a   b     c     d
    0  0  10  20   NaN   NaN
    1  1  11  21   NaN   NaN
    2  2  12  22  32.0  42.0
    3  3  13  23  33.0  43.0
         i     a     b   c   d
    2.0  2  12.0  22.0  32  42
    3.0  3  13.0  23.0  33  43
    NaN  4   NaN   NaN  34  44
    NaN  5   NaN   NaN  35  45
         i     a     b   c   d
    2.0  2  12.0  22.0  32  42
    3.0  3  13.0  23.0  33  43
    NaN  4   NaN   NaN  34  44
    NaN  5   NaN   NaN  35  45
    


```python
print(pd.merge(df3, df4, left_on='i', right_index=True, how='inner'))
print(df3.join(df4, on='i', how='inner'))
print(pd.merge(df3, df4, left_on='i', right_index=True, how='outer'))
print(df3.join(df4, on='i', how='outer')) 
```

       i   a   b   c   d
    2  2  12  22  32  42
    3  3  13  23  33  43
       i   a   b   c   d
    2  2  12  22  32  42
    3  3  13  23  33  43
         i     a     b     c     d
    0.0  0  10.0  20.0   NaN   NaN
    1.0  1  11.0  21.0   NaN   NaN
    2.0  2  12.0  22.0  32.0  42.0
    3.0  3  13.0  23.0  33.0  43.0
    NaN  4   NaN   NaN  34.0  44.0
    NaN  5   NaN   NaN  35.0  45.0
         i     a     b     c     d
    0.0  0  10.0  20.0   NaN   NaN
    1.0  1  11.0  21.0   NaN   NaN
    2.0  2  12.0  22.0  32.0  42.0
    3.0  3  13.0  23.0  33.0  43.0
    NaN  4   NaN   NaN  34.0  44.0
    NaN  5   NaN   NaN  35.0  45.0
    

# 파일 입출력


```python
x = {'subj': ['math', 'comp', 'phys', 'chem', 'biol'],
    'score': [100, 95, 85, 75, 80],
    'avg': [95.2, 66.1, 69.5, 86.8, 91.2]}
df1 = pd.DataFrame(x, index=list('abcde'))
df2 = df1.reindex(list('abcdef'))

print(df2) 
```

       subj  score   avg
    a  math  100.0  95.2
    b  comp   95.0  66.1
    c  phys   85.0  69.5
    d  chem   75.0  86.8
    e  biol   80.0  91.2
    f   NaN    NaN   NaN
    


```python
df2.to_csv('./df2.csv', sep=',', na_rep='NaN')
df2.to_csv('./df2_ind.csv', sep=',', na_rep='NaN', index=False) 
```


```python
## 이 코드는 윈도우 환경에서만 가능함
# import win32com.client
# def openEx(fn):
#     """ 엑셀 어플리케이션 선언 """
#     excel = win32com.client.Dispatch("Excel.Application")
#     """ 엑셀 프로그램 보여지게 함 """
#     excel.Visible = True
#     """ 워크북 을 읽어 들인다. """
#     wb = excel.Workbooks.Open(fn)

# import os
# openEx(os.getcwd() + '/df2.csv')
# openEx(os.getcwd() + '/df2_ind.csv')
```


```python
df = pd.read_csv('df3.csv', index_col=0, sep='|')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subj</th>
      <th>score</th>
      <th>avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>math</td>
      <td>100.0</td>
      <td>95.2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>comp</td>
      <td>95.0</td>
      <td>66.1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>phys</td>
      <td>85.0</td>
      <td>69.5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>chem</td>
      <td>75.0</td>
      <td>86.8</td>
    </tr>
    <tr>
      <th>e</th>
      <td>biol</td>
      <td>80.0</td>
      <td>91.2</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.read_csv('./df2_ind.csv')
df3 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subj</th>
      <th>score</th>
      <th>avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>math</td>
      <td>100.0</td>
      <td>95.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comp</td>
      <td>95.0</td>
      <td>66.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phys</td>
      <td>85.0</td>
      <td>69.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chem</td>
      <td>75.0</td>
      <td>86.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>biol</td>
      <td>80.0</td>
      <td>91.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.to_csv('./df2_ind_sep.csv', sep='|', na_rep='NaN', index=False)
df4 = pd.read_csv('./df2_ind_sep.csv', sep='|')
df4 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subj</th>
      <th>score</th>
      <th>avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>math</td>
      <td>100.0</td>
      <td>95.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comp</td>
      <td>95.0</td>
      <td>66.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phys</td>
      <td>85.0</td>
      <td>69.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chem</td>
      <td>75.0</td>
      <td>86.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>biol</td>
      <td>80.0</td>
      <td>91.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = [['math', 'comp', 'phys', 'chem', 'biol'],
    [100, 95, 85, 75, 80],
    [95.2, 66.1, 69.5, 86.8, 91.2]]
df5 = pd.DataFrame(y, index=list('abc'))
df5.to_csv('./df5.csv', sep=',', index=False, header=None)
df6 = pd.read_csv('./df5.csv', header=None, names=list('ABCDE'))
df6 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>math</td>
      <td>comp</td>
      <td>phys</td>
      <td>chem</td>
      <td>biol</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>95</td>
      <td>85</td>
      <td>75</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>95.2</td>
      <td>66.1</td>
      <td>69.5</td>
      <td>86.8</td>
      <td>91.2</td>
    </tr>
  </tbody>
</table>
</div>


