# 아래 링크를 통해 주피터 노트북 뷰어(nbviewer.jupyter.org)로 보거나 구글 코랩(colab.research.google.com)에서 실행할 수 있음

<pre>
    <a target="_blank" href="https://nbviewer.jupyter.org/github/KimDaeHo26/python/blob/main/Numpy.ipynb"><img src="https://jupyter.org/assets/main-logo.svg" width="28" />주피터 노트북 뷰어로 보기</a>
<br>
    <a target="_blank" href="https://colab.research.google.com/github/KimDaeHo26/python/blob/main/Numpy.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
</pre>

## numpy  인공지능에서 주로 사용되는 벡터 및 행렬 연산에서 매우 편리한 기능을 제공


```python
import numpy as np  
np.__version__
```




    '1.19.2'




```python
lst1 = [1, 2, 3, 4, 5]
print(type(lst1), lst1)
arr1 = np.array(lst1)
print(type(arr1), arr1) 
lst2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
arr2 = np.array(lst2)
print(type(arr1), arr2)
```

    <class 'list'> [1, 2, 3, 4, 5]
    <class 'numpy.ndarray'> [1 2 3 4 5]
    <class 'numpy.ndarray'> [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    int32 int32
    

## 크기 / 차원


```python
print('1차원:', np.ndim(arr1)) 
print('2차원:', np.ndim(arr2))

print('원소개수:', np.size(arr1))
print('원소개수:', np.size(arr2))

print('차원 몇 원소개수:', np.shape(arr1))  
print('차원 몇 원소개수:', np.shape(arr2))
```

    1차원: 1
    2차원: 2
    원소개수: 5
    원소개수: 12
    차원 몇 원소개수: (5,)
    차원 몇 원소개수: (4, 3)
    


```python
print(arr2.shape)
tmp = arr2.reshape(3,4)  # 배열 크기 변경
print(tmp.shape)
tmp2 = arr2.reshape(2,-1)  # 배열 크기 변경 (-1 은 나머지 값을 자동으로 넣어줌)
print(tmp2.shape)
```

    (4, 3)
    (3, 4)
    (2, 6)
    


```python
print(arr1.dtype)
print(arr2.dtype)
```

    int32
    int32
    


```python
print(arr1.astype(float))     # 데이터 타입으로 바꾸고 싶을 때에는 astype 함수를 이용
print(arr1.astype(complex)) 
print(arr1.astype(np.string_))
```

    [1. 2. 3. 4. 5.]
    [1.+0.j 2.+0.j 3.+0.j 4.+0.j 5.+0.j]
    [b'1' b'2' b'3' b'4' b'5']
    

## 배열 생성 함수


```python
arr3 = np.array((1,2))
print(arr3)
arr4 = np.asfarray([1,2])
print('실수형 asfarray:',arr4)
 
print(arr3.dtype, arr4.dtype)
```

    [1 2]
    실수형 asfarray: [1. 2.]
    int32 float64
    


```python
# NaN이란 Not a Number라는 뜻으로, 숫자가 아님을 뜻하는 상수입니다.
# INF는 Infinite라는 뜻으로, 무한대 값을 뜻하는 상수
arr4 = np.array([1, 2, 3, 4])
arr5 = np.array([1, 2, np.nan, 4])
print(np.asarray_chkfinite(arr4))  # chkfinite는 check finite의 줄임말로 np.nan이나 np.inf가 배열에 들어있는지 확인 
try :
    print(np.asarray_chkfinite(arr5)) 
except ValueError as e :
    print(arr5)
    print(e)
    
```

    [1 2 3 4]
    [ 1.  2. nan  4.]
    array must not contain infs or NaNs
    


```python
print('0~4:', np.arange(5))
print('3~9:', np.arange(3, 10))
print('2~9 2자리씩 건너 띄워서:', np.arange(2, 10, 2)) 
```

    0~4: [0 1 2 3 4]
    3~9: [3 4 5 6 7 8 9]
    2~9 2자리씩 건너 띄워서: [2 4 6 8]
    

## linspace 선형 간격


```python
print('0~1 4개로 나눠서:', np.linspace(0, 5, 4))
```

    0~1 4개로 나눠서: [0.         1.66666667 3.33333333 5.        ]
    

## zeros :   0으로 채워줌


```python
print(np.zeros(5))
print(np.zeros((2, 3)))
```

    [0. 0. 0. 0. 0.]
    [[0. 0. 0.]
     [0. 0. 0.]]
    

## ones : 1로 채워줌


```python
print(np.ones(3))
print(np.ones((2, 3)))
```

    [1. 1. 1.]
    [[1. 1. 1.]
     [1. 1. 1.]]
    

## full : 주어진 값으로 채워줌


```python
print(np.full(3, 4))
print(np.full(2, 4), 5)
```

    [4 4 4]
    [4 4] 5
    

## eye, identity : 항등행렬을 만들어줌


```python
print(np.eye(3))
print(np.identity(4)) 
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    

## diag : 주어진 배열 값을 이용하여 대각행렬을 만들어줌


```python
print(np.diag(np.array([1, 2, 3, 4])))
print(np.diag(1+np.arange(4)))
```

    [[1 0 0 0]
     [0 2 0 0]
     [0 0 3 0]
     [0 0 0 4]]
    [[1 0 0 0]
     [0 2 0 0]
     [0 0 3 0]
     [0 0 0 4]]
    

## np.where


```python
a = np.array([[0, 1, 2],
              [0, 2, 4],
              [0, 3, 6]])
print( np.where(a < 4, a, -1) ) # a가 4 보다 작으면 a 아니면 -1
```

    [[ 0  1  2]
     [ 0  2 -1]
     [ 0  3 -1]]
    

## arange : range 와 유사


```python
arr6 = np.arange(12)
print(arr6)
arr7 = np.arange(12).reshape(3, 4)   #reshape 함수는 1차원 배열을 다차원 배열로 바꿔주는 함수 
print(arr7) 
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
arr8 = np.arange(12)
print(arr8.reshape(3, -1))   # 전체 길이를 모른 상태로 한 쪽 길이만 지정해주고 싶다면 -1을 사용 
print(arr8.reshape(4, -1))
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    

# 배열 인덱싱/슬라이싱


```python
lst1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
arr1 = np.array(lst1) 
print(arr1)
```

    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    


```python
print('2줄~3줄, 1컬럼~2컬럼:', arr1[1:3, 0:2], sep='\n')
```

    2줄~3줄, 1컬럼~2컬럼:
    [[4 5]
     [7 8]]
    


```python
print('2줄~끝, 3컬럼~끝:', arr1[1:, 2:], sep='\n')
```

    2줄~끝, 3컬럼~끝:
    [[ 6]
     [ 9]
     [12]]
    


```python
print('첫번째줄:', arr1[0])
print('첫번째줄 전체컬럼:', arr1[0, ])
print('첫번째줄 전체컬럼:', arr1[0, :]) 
```

    첫번째줄: [1 2 3]
    첫번째줄 전체컬럼: [1 2 3]
    첫번째줄 전체컬럼: [1 2 3]
    


```python
print('전체줄 첫번째 컬럼:', arr1[:, 0])
```

    전체줄 첫번째 컬럼: [ 1  4  7 10]
    


```python
arr2 = np.arange(7)
arr3 = arr2[1:5]   # 주소가 복사 됨
print(arr2)
print(arr3)

arr3[2] = 10
print(arr2)
print(arr3) 
```

    [0 1 2 3 4 5 6]
    [1 2 3 4]
    [ 0  1  2 10  4  5  6]
    [ 1  2 10  4]
    


```python
arr2 = np.arange(7)
arr3 = arr2[1:5].copy()   # 값이 복사 됨
print(arr2)
print(arr3)

arr3[2] = 10
print(arr2)
print(arr3) 
```

    [0 1 2 3 4 5 6]
    [1 2 3 4]
    [0 1 2 3 4 5 6]
    [ 1  2 10  4]
    


```python
lst1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
arr1 = np.array(lst1) 
print('arr1 :', arr1)
print('arr1[0, 2] :', arr1[0, 2])
print('arr1[[0, 2], [1, 2]] :', arr1[[0, 2], [1, 2]])   # [0, 2], [1, 2] 이므로 하나씩 , 즉, (0, 1)과 (2, 2) 
```

    arr1 : [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    arr1[0, 2] : 3
    arr1[[0, 2], [1, 2]] : [2 9]
    

# 논리형 인덱싱


```python
lst1 = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]

blst = [[False, True, True],
       [True, False, False],
       [True, True, False]]

arr1 = np.array(lst1)
barr1 = np.array(blst)
print(arr1) 
print(barr1) 
print(arr1[barr1])
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    [[False  True  True]
     [ True False False]
     [ True  True False]]
    [2 3 4 7 8]
    


```python
print(arr1[arr1%2==0])
print(arr1[arr1%3==0]) 
```

    [2 4 6 8]
    [3 6 9]
    


```python
subject = np.array(['math', 'math', 'lang', 'chem', 'math', 'phys', 'chem'])
data = np.arange(28).reshape(7, 4)

print(subject)
print(data) 
```

    ['math' 'math' 'lang' 'chem' 'math' 'phys' 'chem']
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]
     [24 25 26 27]]
    


```python
print(subject=='math')   # 논리형을 이용하여 논리형 배열을 생성
```

    [ True  True False False  True False False]
    


```python
print(data[subject=='math', :])
print(data[(subject=='math') | (subject=='chem'), :]) 
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [16 17 18 19]]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [12 13 14 15]
     [16 17 18 19]
     [24 25 26 27]]
    

# 난수 생성


```python
np.random.seed(seed=1234)   
# 난수 생성 함수는 난수를 생성할 때마다 각각 다른 값을 추출하게 됩니다
# 만약에 동일한 값을 추출하고자 할 때는 seed라는 값을 이용
```

## 정규분포 normal


```python
"""정규분포 함수는 normal(loc, scale, size)로 구성
loc은 평균, scale은 표준편차, size는 개수를 의미합니다
설정하지 않는 경우 loc 0, scale 1로 설정됩니다
"""

print(np.random.normal(size=5))
print(np.random.normal(size=5))

np.random.seed(seed=1234)  # seed를 설정한 후에는 처음 시작하는 값이 동일함
print(np.random.normal(size=5))
print(np.random.normal(size=5))
np.random.seed(seed=1234)
print(np.random.normal(size=5)) 
print(np.random.normal(size=5)) 
print(np.random.normal(size=5)) 
np.random.seed(seed=1234)
print(np.random.normal(size=5)) 
print(np.random.normal(size=5)) 
print(np.random.normal(size=5))
```

    [ 0.00211836  0.40545341  0.28909194  1.32115819 -1.54690555]
    [-0.20264632 -0.65596934  0.19342138  0.55343891  1.31815155]
    [ 0.47143516 -1.19097569  1.43270697 -0.3126519  -0.72058873]
    [ 0.88716294  0.85958841 -0.6365235   0.01569637 -2.24268495]
    [ 0.47143516 -1.19097569  1.43270697 -0.3126519  -0.72058873]
    [ 0.88716294  0.85958841 -0.6365235   0.01569637 -2.24268495]
    [ 1.15003572  0.99194602  0.95332413 -2.02125482 -0.33407737]
    [ 0.47143516 -1.19097569  1.43270697 -0.3126519  -0.72058873]
    [ 0.88716294  0.85958841 -0.6365235   0.01569637 -2.24268495]
    [ 1.15003572  0.99194602  0.95332413 -2.02125482 -0.33407737]
    


```python
print(np.random.normal(size=(2, 3))) 
```

    [[ 0.00211836  0.40545341  0.28909194]
     [ 1.32115819 -1.54690555 -0.20264632]]
    

## 표준정규분포 함수는 randn


```python
print(np.random.randn(10))
print(np.random.randn(5, 4)) 
```

    [-0.65596934  0.19342138  0.55343891  1.31815155 -0.46930528  0.67555409
     -1.81702723 -0.18310854  1.05896919 -0.39784023]
    [[ 0.33743765  1.04757857  1.04593826  0.86371729]
     [-0.12209157  0.12471295 -0.32279481  0.84167471]
     [ 2.39096052  0.07619959 -0.56644593  0.03614194]
     [-2.0749776   0.2477922  -0.89715678 -0.13679483]
     [ 0.01828919  0.75541398  0.21526858  0.84100879]]
    

## 이항분포 함수는 binomial(n, p, size)로 구성


```python
print(np.random.binomial(n=1, p=0.5, size=5))
print(np.random.binomial(n=1, p=0.5, size=5))
```

    [0 0 0 0 0]
    [1 0 0 1 1]
    

## 초기하분포 함수는 hypergeometric(ngood, nbad, nsample, size)로 구성


```python
print(np.random.hypergeometric(ngood=5, nbad=10, nsample=10, size=10))
print(np.random.hypergeometric(ngood=5, nbad=10, nsample=10, size=10))
```

    [3 4 4 3 5 3 3 4 4 5]
    [4 3 4 3 3 2 5 2 3 4]
    

## 포아송 분포는 poisson(lam, size)로 구성


```python
print(np.random.poisson(lam=5, size=10))
print(np.random.poisson(lam=5, size=10))
```

    [4 6 4 9 6 4 4 3 8 2]
    [1 6 7 3 4 8 7 2 6 4]
    

## T 분포는 standard_t(df, size)로 구성


```python
print(np.random.standard_t(df=5, size=10))
print(np.random.standard_t(df=5, size=10))
```

    [-1.27227362  0.37843737 -0.31888787  1.69801724 -0.9015829  -2.40668973
     -0.55747795  1.07216807  0.26198781  0.15415054]
    [-0.01924681  0.21414425 -0.19280827 -0.11181867  1.29599052 -1.35898185
     -0.02469016  1.27948939  1.13315027  1.43719214]
    

## F 분포는 f(dfnum, dfden, size)로 구성


```python
print(np.random.f(dfnum=5, dfden=10, size=10))
print(np.random.f(dfnum=5, dfden=10, size=10))
```

    [0.25702863 0.84891219 1.669646   0.10106778 0.25174516 0.26426942
     0.47087418 0.17313482 0.79345206 0.93234207]
    [0.46940954 3.0694736  0.39116676 0.36801559 0.41804066 0.98598247
     0.45089244 3.13491733 0.67145314 0.9746113 ]
    

## 균등분포는 uniform(low, high, size)로 구성


```python
print(np.random.uniform(low=0.0, high=5.0, size=7))
print(np.random.uniform(low=0.0, high=5.0, size=7))
```

    [4.85462662 3.86930482 0.65888236 4.63589932 4.1389829  1.96365833
     0.70285323]
    [0.2606817  0.46364658 4.43494815 0.84075484 1.81443306 3.27140943
     3.02410977]
    

## 이산형 균등분포는 randint(low, high, size)로 구성


```python
print(np.random.randint(low=0, high=10, size=10))
print(np.random.randint(low=0, high=10, size=10))
```

    [1 0 1 1 8 3 5 4 4 2]
    [4 5 6 4 2 6 1 7 6 2]
    

## 카이제곱분포는 chisquare(df, size)로 구성


```python
print(np.random.chisquare(df=5, size=10))
print(np.random.chisquare(df=5, size=10))
```

    [3.47707365 2.81074118 3.85110115 2.61522187 6.35041136 3.72485109
     5.38442788 5.16531962 5.20909845 0.96986269]
    [ 5.18328966  2.82207206  4.4101532   2.85744122  8.47098117 13.76316205
      1.26125452  5.66785694  6.80769451  3.01881466]
    

## 감마분포는 gamma(shape, scale, size)로 구성


```python
print(np.random.gamma(shape=4, scale=4, size=10))
print(np.random.gamma(shape=4, scale=4, size=10))
```

    [ 9.94292627 11.18724428 11.72346153  9.54689186 15.44508413 20.956999
     22.87095096 10.75768702  5.82416155 13.43643477]
    [15.59805859 19.05329372 21.90175984  9.90535727 32.31267163 12.12177899
     14.4330887  32.06832125 28.63681561 17.61792856]
    

## 순서바꾸기


```python
x = np.arange(7)
print(x)

print(np.random.permutation(x))  # permutation은 해당 함수를 실행한 순간 단 한 번 바꿔주는 함수
print(x)

print(np.random.shuffle(x)) # shuffle은 바꿔준 후에도 바뀐 순서가 유지되는 함수
print(x)
```

    [0 1 2 3 4 5 6]
    [0 5 4 3 2 1 6]
    [0 1 2 3 4 5 6]
    None
    [0 3 1 6 4 2 5]
    

# 배열 연산


```python
x = np.array([1, 2, 3, 4, 5])
print(x+1)
print(x-1)
print(x*2)
print(x/2)
print(x//2)
print(x%2)
print(x**2)
print(2**x)
```

    [2 3 4 5 6]
    [0 1 2 3 4]
    [ 2  4  6  8 10]
    [0.5 1.  1.5 2.  2.5]
    [0 1 1 2 2]
    [1 0 1 0 1]
    [ 1  4  9 16 25]
    [ 2  4  8 16 32]
    


```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x//y)
print(x%y)
print(x**y)
```

    [ 3  5  7  9 11]
    [-1 -1 -1 -1 -1]
    [ 2  6 12 20 30]
    [0.5        0.66666667 0.75       0.8        0.83333333]
    [0 0 0 0 0]
    [1 2 3 4 5]
    [    1     8    81  1024 15625]
    


```python
# 연산 함수 이용
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
print(np.add(x, y))
print(np.subtract(x, y))
print(np.multiply(x, y))
print(np.divide(x, y))
print(np.power(x, y))
print(np.mod(x, y))
```

    [ 3  5  7  9 11]
    [-1 -1 -1 -1 -1]
    [ 2  6 12 20 30]
    [0.5        0.66666667 0.75       0.8        0.83333333]
    [    1     8    81  1024 15625]
    [1 2 3 4 5]
    


```python
z = np.array([-1, 3, -5])
print(np.abs(z))
print(np.fabs(z))
```

    [1 3 5]
    [1. 3. 5.]
    


```python
y = np.array([2, 3, 4, 5, 6])
z = np.array([-1, 3, -5])
print(np.sqrt(y))
print(np.square(y))
print(np.exp(y))
print(np.log(y))
print(np.log10(y))
print(np.log2(y))
print(np.sign(z))
```

    [1.41421356 1.73205081 2.         2.23606798 2.44948974]
    [ 4  9 16 25 36]
    [  7.3890561   20.08553692  54.59815003 148.4131591  403.42879349]
    [0.69314718 1.09861229 1.38629436 1.60943791 1.79175947]
    [0.30103    0.47712125 0.60205999 0.69897    0.77815125]
    [1.         1.5849625  2.         2.32192809 2.5849625 ]
    [-1  1 -1]
    


```python
x = np.array([1, 2, 5, 10, 35, 98])
print(np.diff(x))
print(np.diff(x, n=2))
print(np.diff(x, n=3))
# diff 함수 안에 n=2를 넣게 된다면, 차이에 대한 차이를 또 계산
```

    [ 1  3  5 25 63]
    [ 2  2 20 38]
    [ 0 18 18]
    


```python
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2], [3, 4]])

print(np.prod(a))   # prod 함수는 배열의 요소들의 곱을 구해주는 함수
print(np.prod(b, axis=0))
print(np.prod(b, axis=1))
"""axis=0이면 1차원들의 곱이 되어 1*3=3, 2*4=8
axis=1이면 2차원들의 곱이 되어 1*2=2, 3*4=12"""
```

    24
    [3 8]
    [ 2 12]
    




    'axis=0이면 1차원들의 곱이 되어 1*3=3, 2*4=8\naxis=1이면 2차원들의 곱이 되어 1*2=2, 3*4=12'




```python
a = np.random.randint(0, 100, 5)
b = np.random.randint(0, 100, 5)

print(a)
print(b)
print(np.maximum(a, b))
print(np.minimum(a, b))
```

    [88 29 61 89 46]
    [41 97 15 42 96]
    [88 97 61 89 96]
    [41 29 15 42 46]
    

# 수학 함수


```python
a = np.array([3.14, -5.21, 9.69, -1.87, 0])
print(np.ceil(a))
print(np.floor(a))
print(np.rint(a))
print(np.around(a))
# ceil은 소수점 올림 함수, floor은 소수점 내림 함수, rint와 round는 소수점 반올림 함수
```

    [ 4. -5. 10. -1.  0.]
    [ 3. -6.  9. -2.  0.]
    [ 3. -5. 10. -2.  0.]
    [ 3. -5. 10. -2.  0.]
    


```python
print(np.round_(a, 1))
print(np.round_(a, -1))
print(np.fix(a))
print(np.trunc(a))
"""
round_ 함수는 주어진 소수점 자릿수까지 반올림하는 함수입니다.
1이면 소수점 1자리까지, -1이면 십의 자리까지 반올림을 진행합니다
fix 함수는 0의 방향으로 가까운 정수로 올림 또는 내림을 진행하는 함수입니다
trunc 함수는 소수점을 떼는 함수입니다.
"""
```

    [ 3.1 -5.2  9.7 -1.9  0. ]
    [  0. -10.  10.  -0.   0.]
    [ 3. -5.  9. -1.  0.]
    [ 3. -5.  9. -1.  0.]
    




    '\nround_ 함수는 주어진 소수점 자릿수까지 반올림하는 함수입니다.\n1이면 소수점 1자리까지, -1이면 십의 자리까지 반올림을 진행합니다\nfix 함수는 0의 방향으로 가까운 정수로 올림 또는 내림을 진행하는 함수입니다\ntrunc 함수는 소수점을 떼는 함수입니다.\n'




```python
print(np.sin(90))
print(np.cos(90))
print(np.tan(90))
print(np.sinh(90))
print(np.cosh(90))
print(np.tanh(90))
# sin, cos, tan은 각각 삼각함수의 값을 리턴해줍니다
# sinh, cosh, tanh는 하이퍼볼릭 삼각함수의 값을 리턴해줍니다 
```

    0.8939966636005579
    -0.4480736161291701
    -1.995200412208242
    6.102016471589204e+38
    6.102016471589204e+38
    1.0
    


```python
print(np.arcsin(0.5))
print(np.arcsinh(0.5))
print(np.arccos(0.5))
print(np.arccosh(1.5))
print(np.arctan(0.5))
print(np.arctanh(0.5)) 
# arcsin, arccos, arctan은 각각 역삼각함수 값을 리턴해줍니다
# arcsinh, arccosh, arctanh는 하이퍼볼릭 역삼각함수 값을 리턴해줍니다
```

    0.5235987755982989
    0.48121182505960347
    1.0471975511965979
    0.9624236501192069
    0.4636476090008061
    0.5493061443340549
    


```python
print(np.deg2rad(180))
print(np.rad2deg(np.pi)) 
# deg2rad 함수는 디그리를 라디안으로, rad2deg는 라디안을 디그리로 변환해줍니다
# np.pi는 파이 상수를 저장하는 numpy 라이브러리의 상수입니다.
```

    3.141592653589793
    180.0
    


```python
a = np.random.randn(5, 4)
print(a.sum())
print(np.sum(a))
print(a.mean())
print(np.mean(a))
# sum은 전체 원소의 합, mean은 평균 값
```

    -2.9367065730457425
    -2.9367065730457425
    -0.14683532865228713
    -0.14683532865228713
    


```python
a = np.random.randn(5, 4)
print(a)
print(np.corrcoef(a))
print(np.std(a))
print(np.var(a))
# corrcoef 함수는 상관계수, std 함수는 표준편차, var 함수는 분산 값
```

    [[ 4.71435164e-01 -1.19097569e+00  1.43270697e+00 -3.12651896e-01]
     [-7.20588733e-01  8.87162940e-01  8.59588414e-01 -6.36523504e-01]
     [ 1.56963721e-02 -2.24268495e+00  1.15003572e+00  9.91946022e-01]
     [ 9.53324128e-01 -2.02125482e+00 -3.34077366e-01  2.11836468e-03]
     [ 4.05453412e-01  2.89091941e-01  1.32115819e+00 -1.54690555e+00]]
    [[ 1.         -0.00158431  0.76726132  0.60503006  0.53994372]
     [-0.00158431  1.         -0.38880856 -0.78881739  0.63075989]
     [ 0.76726132 -0.38880856  1.          0.71084599 -0.11480404]
     [ 0.60503006 -0.78881739  0.71084599  1.         -0.10698581]
     [ 0.53994372  0.63075989 -0.11480404 -0.10698581  1.        ]]
    1.0739542937435573
    1.1533778250502231
    


```python
a = np.random.randn(5, 4)
print(np.min(a))
print(np.max(a))
print(np.argmin(a))
print(np.argmax(a))
# min, max 함수는 각각 전체에서의 최솟값, 최댓값
# argmin, argmax 함수는 각각 값들이 위치한 인덱스 위치를 반환
```

    -1.8170272265901968
    2.390960515463033
    7
    19
    


```python
a = np.random.randn(5, 4)
print(a)
print('누적합:' ,np.cumsum(a))
print('누적곱:', np.cumprod(a))
```

    [[ 0.46439233 -3.56351666  1.32110562  0.15263055]
     [ 0.16452954 -0.43009569  0.76736874  0.98491984]
     [ 0.27083585  1.39198619  0.07984231 -0.39996458]
     [-1.02785056 -0.58471821  0.81659393 -0.08194705]
     [-0.34476601  0.52828815 -1.06898878 -0.51188131]]
    누적합: [ 0.46439233 -3.09912434 -1.77801872 -1.62538817 -1.46085862 -1.89095432
     -1.12358558 -0.13866574  0.13217011  1.5241563   1.60399862  1.20403404
      0.17618348 -0.40853473  0.40805919  0.32611214 -0.01865387  0.50963427
     -0.55935451 -1.07123582]
    누적곱: [ 4.64392325e-01 -1.65486979e+00 -2.18625777e+00 -3.33689731e-01
     -5.49018188e-02  2.36130357e-02  1.81199054e-02  1.78466543e-02
      4.83351377e-03  6.72818444e-03  5.37193808e-04 -2.14858496e-04
      2.20842425e-04 -1.29130588e-04 -1.05447254e-04  8.64109157e-06
     -2.97915470e-06 -1.57385211e-06  1.68243025e-06 -8.61204600e-07]
    


```python
print(a)
print('행방향합:', a.sum(axis=0))
print('행방향합:', np.sum(a, axis=0))
print('열방향합:', a.sum(axis=1))
print('열방향합:', np.sum(a, axis=1))
```

    [[ 0.46439233 -3.56351666  1.32110562  0.15263055]
     [ 0.16452954 -0.43009569  0.76736874  0.98491984]
     [ 0.27083585  1.39198619  0.07984231 -0.39996458]
     [-1.02785056 -0.58471821  0.81659393 -0.08194705]
     [-0.34476601  0.52828815 -1.06898878 -0.51188131]]
    행방향합: [-0.47285886 -2.65805622  1.91592181  0.14375745]
    행방향합: [-0.47285886 -2.65805622  1.91592181  0.14375745]
    열방향합: [-1.62538817  1.48672243  1.34269977 -0.8779219  -1.39734796]
    열방향합: [-1.62538817  1.48672243  1.34269977 -0.8779219  -1.39734796]
    


```python
a = np.array([1, 2, 3, 4])
b = np.sum(a)
c = np.sum(a, keepdims=True) # keepdims가 True인 경우 1차원인 a의 차원을 유지한채로 10

print(b)
print(c)
```

    10
    [10]
    

# 논리 연산과 행렬 연산


```python
a = np.array([0, 1, np.nan, 3, np.inf, np.NINF, np.PINF])
# np.nan은 NaN임을 지정하는 상수
# np.inf는 무한대임을 지정하는 상수,
# p.NINF는 -무한대, 
# np.PINF는 +무한대를 뜻하는 상수입니다.
print(a)
print(np.isnan(a))
print(np.isinf(a))
print(np.isneginf(a))
print(np.isposinf(a))
print(np.isfinite(a))
# isnan : NaN인지 판단, isinf : 무한대 값인지 판단, isneginf : -무한대 값인지 판단,
# isposinf : +무한대 값인지 판단, isfinite : 유한 값인지 판단해주는 함수입니다.
```

    [  0.   1.  nan   3.  inf -inf  inf]
    [False False  True False False False False]
    [False False False False  True  True  True]
    [False False False False False  True False]
    [False False False False  True False  True]
    [ True  True False  True False False False]
    


```python
a = np.array([1, 2, 3, 4, 4])
b = np.array([3, 5, 1, 2, 4])
c = np.array([3, 5, 1, 2, 4])

print(np.equal(a, b))
print(np.not_equal(a, b))
print(np.greater(a, b))
print(np.greater_equal(a, b))
print(np.less(a, b))
print(np.less_equal(a, b))
print(np.array_equal(a, b))
print(np.array_equal(b, c))
```

    [False False False False  True]
    [ True  True  True  True False]
    [False False  True  True False]
    [False False  True  True  True]
    [ True  True False False False]
    [ True  True False False  True]
    False
    True
    


```python
a = np.array([1, 2, 3, 4, 5, 6, 7])

print((a>=3).any())
print((a>=3).all())
# any()  해당 조건에 부합하는 원소가 하나라도 존재하면 True
# all()  모든 원소가 해당 조건에 부합하면 True 
```

    True
    False
    


```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(np.logical_not(a))
print(np.logical_and(a, b))
print(np.logical_or(a, b))
print(np.logical_xor(a, b))
```

    [False False  True  True]
    [ True False False False]
    [ True  True  True False]
    [False  True  True False]
    


```python
a = np.random.randint(0, 10, 4).reshape(2, 2)
b = np.random.randint(0, 10, 4).reshape(2, 2)
print(a)
print(b)
```

    [[6 2]
     [9 8]]
    [[7 8]
     [0 8]]
    

# 행렬의 곱을 구할 때는 dot 함수를 이용


```python
print(a*b)
print(np.multiply(a, b))
print(np.dot(a, b))  # 행렬곱
```

    [[42 16]
     [ 0 64]]
    [[42 16]
     [ 0 64]]
    [[ 42  64]
     [ 63 136]]
    


```python
print(np.diag(np.array([1, 2, 3, 4])))
print(np.diag(1+np.arange(4)))
# diag 함수는 주어진 배열 값을 이용하여 대각행렬을 만들어주는 함수
```

    [[1 0 0 0]
     [0 2 0 0]
     [0 0 3 0]
     [0 0 0 4]]
    [[1 0 0 0]
     [0 2 0 0]
     [0 0 3 0]
     [0 0 0 4]]
    


```python
x = np.arange(16).reshape(4, 4)
print(x)
print(np.trace(x))
# trace는 이름 그대로 대각합을 구하는 함수
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    30
    


```python
y = np.array([[1, 2], [3, 4]])
print(np.linalg.det(y))
# linalg.det 함수는 행렬식(determinant)을 구해주는 함수
```

    -2.0000000000000004
    


```python
a = np.array([[1, 2], [3, 4]])
b = np.array([9, 8])
x = np.linalg.solve(a, b)
print(x)
# linalg.solve 함수는 주어진 연립방정식을 해결해주는 함수입니다.
# 위의 예를 들면 1x+3y=9, 2x+4y=8인 식에 대해서 x=-10, y=9.5를 답으로 출력
```

    [-10.    9.5]
    


```python
a = np.array([[1, 2], [3, 4], [6, 5]])
print(a)
print(a.T)
print(a.swapaxes(0,1))
print(a.transpose(1,0))
# T, swapaxes, transpose 함수는 축 순서를 바꿔주는 함수입니다
# T는 전치행렬을 만드는 함수, swapaxes는 두 축을 서로 바꿔주는 함수, transpose는 지정된 축 순서대로 배열을 재배치 하는 함수
```

    [[1 2]
     [3 4]
     [6 5]]
    [[1 3 6]
     [2 4 5]]
    [[1 3 6]
     [2 4 5]]
    [[1 3 6]
     [2 4 5]]
    

## 정렬


```python
a = np.random.randint(0, 10, 5)
b = np.random.randint(0, 30, (4, 3))

print(a)
print('정렬:', np.sort(a))
print(b)
print('행정렬', np.sort(b, axis=0))
print('열정렬', np.sort(b, axis=1)) 
```

    [0 7 0 2 8]
    정렬: [0 0 2 7 8]
    [[14 20  8]
     [19 13  2]
     [28 20  7]
     [23 10 27]]
    행정렬 [[14 10  2]
     [19 13  7]
     [23 20  8]
     [28 20 27]]
    열정렬 [[ 8 14 20]
     [ 2 13 19]
     [ 7 20 28]
     [10 23 27]]
    

## 중복제거


```python
a = np.array([1, 1, 3, 3, 2, 1, 2, 3, 1, 2, 3])
print(np.unique(a))
```

    [1 2 3]
    


```python
x = np.array([[1, 2], [3, 4], [5, 5]])
print(x.shape)
print(x)

x2 = x.flatten()   # 차원을 하나 줄여서 눌러준다고 하여 flatten 함수
print(x2.shape)
print(x2)
```

    (3, 2)
    [[1 2]
     [3 4]
     [5 5]]
    (6,)
    [1 2 3 4 5 5]
    


```python
a = np.array([1, 5, 3, 4, 1])
b = np.array([1, 2, 4, 4, 2])

print('교집합:', np.intersect1d(a, b))
print('합집합:', np.union1d(a, b))
print('앞의 배열의 원소가 뒤 배열의 원소에 포함되는 경우에 True:', np.in1d(a, b))
print('차집합:', np.setdiff1d(a, b))
print('대칭차집합(합집합-교집합):', np.setxor1d(a, b))
```

    교집합: [1 4]
    합집합: [1 2 3 4 5]
    앞의 배열의 원소가 뒤 배열의 원소에 포함되는 경우에 True: [ True False False  True  True]
    차집합: [3 5]
    대칭차집합(합집합-교집합): [2 3 5]
    

# 브로드캐스팅

일반적으로 numpy의 배열은 크기가 다른 배열은 연산이 불가능합니다
그러나 이를 가능하게 해주는 것이 브로드캐스팅입니다
브로드캐스팅은 연산이 특정 조건이 만족할 경우에 배열의 크기를 자동으로 조절 및 변환하여 연산이 가능하도록 합니다
브로드캐스팅은 특정 차원의 크기가 1일때만 가능하며, 차원에 대하여 축의 길이가 동일해야 합니다 


```python
a = np.arange(12).reshape((4, 3))
b = np.arange(3)
print(a)
print(b)
print(a.shape)
print(b.shape)
print(a+b)
```

    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    [0 1 2]
    (4, 3)
    (3,)
    [[ 0  2  4]
     [ 3  5  7]
     [ 6  8 10]
     [ 9 11 13]]
    


```python
a = np.arange(12).reshape((4, 3))
c = np.arange(4).reshape((4,1))
print(a)
print(c)
print(a+c)
```

    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    [[0]
     [1]
     [2]
     [3]]
    [[ 0  1  2]
     [ 4  5  6]
     [ 8  9 10]
     [12 13 14]]
    


```python
a = np.arange(3)
b = np.arange(3)+3
c = np.arange(6).reshape((2, 3))
d = np.arange(6).reshape((2, 3))+6

print('hstack:', np.hstack((a, b)))
print('r_:', np.r_[a, b])
print('concatenate:', np.concatenate((a, b)))
print('concatenate axis=1:', np.concatenate((c, d), axis=1))
# hstack 함수는 horizontal stack 함수로 두 배열을 가로 방향으로 즉, 왼쪽에서 오른쪽으로 붙이는 함수입니다
# r_ 함수는 두 배열을 row 방향으로 붙이는 함수입니다
# concatenate 함수는 두 배열을 붙이는 함수인데, 기본적으로 axis=0이며, axis를 추가하여 설정
```

    hstack: [0 1 2 3 4 5]
    r_: [0 1 2 3 4 5]
    concatenate: [0 1 2 3 4 5]
    concatenate axis=1: [[ 0  1  2  6  7  8]
     [ 3  4  5  9 10 11]]
    


```python
print(np.vstack((a, b)))
print(np.r_[[a], [b]])
print(np.concatenate((c, d)))
```

    [[0 1 2]
     [3 4 5]]
    [[0 1 2]
     [3 4 5]]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    


```python
a = np.arange(4)
print(a)
print(a.shape)
a2 = a[:, np.newaxis]
print(a2)
print(a2.shape)
print(a.reshape((4,1)))
```

    [0 1 2 3]
    (4,)
    [[0]
     [1]
     [2]
     [3]]
    (4, 1)
    [[0]
     [1]
     [2]
     [3]]
    


```python
b = np.arange(6).reshape(2, 3)
print(b)
print(b.shape)

b2 = b[:, :, np.newaxis]
print(b2)
print(b2.shape)

b3 = b[:, np.newaxis, :]
print(b3)
print(b3.shape)
```

    [[0 1 2]
     [3 4 5]]
    (2, 3)
    [[[0]
      [1]
      [2]]
    
     [[3]
      [4]
      [5]]]
    (2, 3, 1)
    [[[0 1 2]]
    
     [[3 4 5]]]
    (2, 1, 3)
    


```python
c = np.arange(3)
print(c)
print(c.shape)
print('-'*10)
c2 = np.tile(c, 2)
print(c2)
print(c2.shape)
print('-'*10)
c3 = np.tile(c, (3, 2))
print(c3)
print(c3.shape)
```

    [0 1 2]
    (3,)
    ----------
    [0 1 2 0 1 2]
    (6,)
    ----------
    [[0 1 2 0 1 2]
     [0 1 2 0 1 2]
     [0 1 2 0 1 2]]
    (3, 6)
    


```python
d = np.arange(6).reshape((2, 3))
print(d)
print(d.shape)

d2 = np.tile(d, (2, 2))
print(d2)
print(d2.shape)
```

    [[0 1 2]
     [3 4 5]]
    (2, 3)
    [[0 1 2 0 1 2]
     [3 4 5 3 4 5]
     [0 1 2 0 1 2]
     [3 4 5 3 4 5]]
    (4, 6)
    


```python
e = np.arange(12).reshape((3, 4))
print(e)

e2 = e.copy()
e2.resize(6, 2)
print(e2)

e3 = e.copy()
e3.reshape(6, 2)
print(e3)
"""resize는 한 번 적용하면 계속 적용되지만, reshape 함수는 함수를 실행한 단 한 번만 적용"""
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [[ 0  1]
     [ 2  3]
     [ 4  5]
     [ 6  7]
     [ 8  9]
     [10 11]]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    




    'resize는 한 번 적용하면 계속 적용되지만, reshape 함수는 함수를 실행한 단 한 번만 적용'




```python
x = np.arange(9).reshape(3, 3)
y = np.arange(9).reshape(3, 3)+9

print(np.append(x, y))
print(np.append(x, y, axis=0))
print(np.append(x, y, axis=1))
# 1차원화 되는 것을 방지하기 위해서 axis를 지정하게 되는데 axis=0을 하게되면 axis 0축에, 1이면 1축에 더해줍
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]
     [15 16 17]]
    [[ 0  1  2  9 10 11]
     [ 3  4  5 12 13 14]
     [ 6  7  8 15 16 17]]
    


```python
x = np.arange(9).reshape(3, 3)
y = np.arange(12).reshape(4, 3)

print(np.append(x, y))
print(np.append(x, y, axis=0))
# print(np.append(x, y, axis=1))
```

    [ 0  1  2  3  4  5  6  7  8  0  1  2  3  4  5  6  7  8  9 10 11]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    


```python
x = np.arange(9).reshape(3, 3)

print(x)
print(np.insert(x, 1, 10))
print(np.insert(x, 1, 10, axis=0))
print(np.insert(x, 1, 10, axis=1))
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    [ 0 10  1  2  3  4  5  6  7  8]
    [[ 0  1  2]
     [10 10 10]
     [ 3  4  5]
     [ 6  7  8]]
    [[ 0 10  1  2]
     [ 3 10  4  5]
     [ 6 10  7  8]]
    


```python
x = np.arange(9).reshape(3, 3)

print(x)
print(np.delete(x, 1))
print(np.delete(x, 1, axis=0))
print(np.delete(x, 1, axis=1))
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    [0 2 3 4 5 6 7 8]
    [[0 1 2]
     [6 7 8]]
    [[0 2]
     [3 5]
     [6 8]]
    


```python
x = np.arange(24).reshape(4, 6)

print(x,'\n') 
print(np.hsplit(x, 2),'\n') 
print(np.hsplit(x, 3),'\n') 
print(np.hsplit(x, [1, 3, 5]))
```

    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]] 
    
    [array([[ 0,  1,  2],
           [ 6,  7,  8],
           [12, 13, 14],
           [18, 19, 20]]), array([[ 3,  4,  5],
           [ 9, 10, 11],
           [15, 16, 17],
           [21, 22, 23]])] 
    
    [array([[ 0,  1],
           [ 6,  7],
           [12, 13],
           [18, 19]]), array([[ 2,  3],
           [ 8,  9],
           [14, 15],
           [20, 21]]), array([[ 4,  5],
           [10, 11],
           [16, 17],
           [22, 23]])] 
    
    [array([[ 0],
           [ 6],
           [12],
           [18]]), array([[ 1,  2],
           [ 7,  8],
           [13, 14],
           [19, 20]]), array([[ 3,  4],
           [ 9, 10],
           [15, 16],
           [21, 22]]), array([[ 5],
           [11],
           [17],
           [23]])]
    


```python
x = np.arange(24).reshape(4, 6)

print(x,'\n')
print(np.vsplit(x, 2),'\n')
print(np.vsplit(x, 4),'\n')
print(np.vsplit(x, [1, 3]),'\n')
```

    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]] 
    
    [array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]]), array([[12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]])] 
    
    [array([[0, 1, 2, 3, 4, 5]]), array([[ 6,  7,  8,  9, 10, 11]]), array([[12, 13, 14, 15, 16, 17]]), array([[18, 19, 20, 21, 22, 23]])] 
    
    [array([[0, 1, 2, 3, 4, 5]]), array([[ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17]]), array([[18, 19, 20, 21, 22, 23]])] 
    
    

# 파일입출력


```python
x = np.array([1, 2, 3, 4])
np.save('./x_file',x)
"""
기본적으로 numpy 배열은 npy, npz, csv, txt, dat 형식을 제공합니다
1개의 배열을 일반적인 방식으로 저장하면 npy로
여러 개의 배열을 압축여부를 선택할 수 있는 npz 파일,
그리고 csv, txt, dat 파일 등이 있습니다.
"""
x2 = np.load('./x_file.npy')
print(x2)
```

    [1 2 3 4]
    


```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

np.savez('./xy_file', x=x, y=y)  # 여러개 파일

xy2 = np.load('./xy_file.npz')

print(xy2)
print(xy2['x'])
print(xy2['y'])
xy2.close()
```

    <numpy.lib.npyio.NpzFile object at 0x00000000052E7D60>
    [1 2 3 4]
    [2 4 6 8]
    


```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

np.savez_compressed('./xy_file_comp', x=x, y=y)  # 압축

xyc = np.load('./xy_file_comp.npz')

print(xyc)
print(xyc['x'])
print(xyc['y'])
```

    <numpy.lib.npyio.NpzFile object at 0x0000000007FC4E80>
    [1 2 3 4]
    [2 4 6 8]
    


```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

np.savetxt('./xy_file.csv', (x, y),
          header='#START',
          footer='#END',
          fmt='%2.2f')
np.savetxt('./xy_file.txt', (x, y),
          header='#START',
          footer='#END',
          fmt='%2.2f')
np.savetxt('./xy_file_del.txt', (x, y),
          delimiter=',',
          header='#START',
          footer='#END',
          fmt='%2.2f')
np.savetxt('./xy_file.dat', (x, y),
          header='#START',
          footer='#END',
          fmt='%2.2f')
```

<pre>
np.savetxt('./xy_file_del.txt', (x, y),
          delimiter=',',
          header='#START',
          footer='#END',
          fmt='%2.2f')
맨 앞에는 파일 이름, 그 뒤에는 저장할 배열 튜플이 있습니다
delimiter는 값 들간을 구분할 구분자를 의미합니다
xy_file_del.txt의 경우에는 delimiter를 ,로 해주었기 때문에 값 사이사이에 쉼표가 있습니다
그러나 다른 파일들의 경우에는 지정하지 않았기 때문에 띄어쓰기로 구분되어있습니다.
header, footer는 전체 값의 맨 앞과 맨 뒤에 들어가는 문자열을 의미합니다
fmt는 데이터가 저장될 포맷을 의미합니다
%2.2f이기 때문에 소수점 두자리 소수로 저장된 것을 확인할 수 있습니다.
