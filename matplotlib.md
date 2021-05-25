**아래 링크를 통해 주피터 노트북 뷰어(nbviewer.jupyter.org)로 보거나 구글 코랩(colab.research.google.com)에서 실행할 수 있음**

<pre>
<a target="_blank" href="https://nbviewer.jupyter.org/github/KimDaeHo26/python/blob/main/matplotlib.ipynb"><img src="https://jupyter.org/assets/main-logo.svg" width="28" />주피터 노트북 뷰어로 보기</a>
<a target="_blank" href="https://colab.research.google.com/github/KimDaeHo26/python/blob/main/matplotlib.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
</pre>

<pre>
예제 파일 : 로컬에 내려 받아서 실행 할 경우 마우스 우클릭 다른 이름으로 저장
<a ref=https://raw.githubusercontent.com/KimDaeHo26/python/main/iris.csv>iris.csv</a>
<a ref=https://raw.githubusercontent.com/KimDaeHo26/python/main/iris.csv>flights.csv</a>
<a ref=https://raw.githubusercontent.com/KimDaeHo26/python/main/iris.csv>tips.csv</a>
<a ref=https://raw.githubusercontent.com/KimDaeHo26/python/main/iris.csv>train.csv</a>
</pre>

# matplotlib


```python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.__version__
```




    '3.3.2'



# matplotlib font 관련


```python
# 사용가능한 폰트 리스트
from matplotlib import font_manager
i = 0
for iter in  font_manager.fontManager.ttflist:
    i = i +1
    if i > 3 : 
        break
    print(iter)

# 폰트리스트가 있는 (json) 파일 위치 : 폰트 새로 설치시 json 파일을 삭제하면 자동으로 새로 생김
import matplotlib as mpl
print(mpl.get_cachedir())

# plt.rc('font',family='Malgun Gothic')  # windows 'Malgun Gothic', Linux 'NanumGothic' ,  max 'AppleGothic'
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트로 지정
```

    <Font 'cmsy10' (cmsy10.ttf) normal normal 400 normal>
    <Font 'DejaVu Sans Mono' (DejaVuSansMono-Bold.ttf) normal normal 700 normal>
    <Font 'cmb10' (cmb10.ttf) normal normal 400 normal>
    C:\Users\User\.matplotlib
    

# 히스토그램


```python
path = ("https://raw.githubusercontent.com/"
        "KimDaeHo26/python/main/{}.csv")

로컬여부 = 'N'
if 로컬여부 == 'Y' :
    iris_path = './iris.csv'
    flights_path = './flights.csv'
    tips_path = './tips.csv'
    train_path = './train.csv'
else :
    iris_path = path.format('iris')
    flights_path = path.format('flights')
    tips_path = path.format('tips')
    train_path = path.format('train')
```


```python
import pandas as pd
# iris = pd.read_csv('./iris.csv') # 로컬 내려 받아 실행 할 경우
iris = pd.read_csv(iris_path)      # 온라인으로 실행 할 경우

print(iris.shape,'\n')
print(iris.head(),'\n')
print(iris.tail(),'\n')
print(iris.describe(),'\n')    # describe는 전체 데이터셋에 대한 통계값
```

    (150, 5) 
    
       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa 
    
         sepal_length  sepal_width  petal_length  petal_width    species
    145           6.7          3.0           5.2          2.3  virginica
    146           6.3          2.5           5.0          1.9  virginica
    147           6.5          3.0           5.2          2.0  virginica
    148           6.2          3.4           5.4          2.3  virginica
    149           5.9          3.0           5.1          1.8  virginica 
    
           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       5.843333     3.057333      3.758000     1.199333
    std        0.828066     0.435866      1.765298     0.762238
    min        4.300000     2.000000      1.000000     0.100000
    25%        5.100000     2.800000      1.600000     0.300000
    50%        5.800000     3.000000      4.350000     1.300000
    75%        6.400000     3.300000      5.100000     1.800000
    max        7.900000     4.400000      6.900000     2.500000 
    
    


```python
plt.hist(iris['sepal_width'], bins=10)   # bins는 전체 막대의 개수
```




    (array([ 4.,  7., 22., 24., 37., 31., 10., 11.,  2.,  2.]),
     array([2.  , 2.24, 2.48, 2.72, 2.96, 3.2 , 3.44, 3.68, 3.92, 4.16, 4.4 ]),
     <BarContainer object of 10 artists>)




    
![png](output_9_1.png)
    



```python
iris.columns = ['꽃받침길이','꽃받침너비','꽃잎길이','꽃잎너비','종']
```


```python
#plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.family'] = 'New Gulim'
plt.hist(iris['꽃받침너비'], bins=10)   # bins는 전체 막대의 개수
plt.xlabel('꽃받침너비')  
plt.ylabel('개수') 
plt.title('꽃받침너비 histogram 히스트그램')  # 제목
plt.show()
```


    
![png](output_11_0.png)
    



```python
iris[(iris.꽃받침너비>=2) & (iris.꽃받침너비<2.5)]['꽃받침너비'].count()
```




    11




```python
plt.hist(iris['꽃받침너비'], bins=10, density=True, # y축을 전체 개수로 하지 않고 밀도로 나타냅니다
         color='green', alpha=0.3)   # color 색깔을 지정, alpha  투명도
plt.xlabel('꽃받침너비')  
plt.ylabel('밀도') 
plt.title('꽃받침너비 histogram 히스트그램')  # 제목
plt.show()
```


    
![png](output_13_0.png)
    



```python
plt.hist(iris['꽃받침너비'], bins=10, density=True,
        color='yellow', alpha=0.3, cumulative=True) # cumulative 누적 
plt.xlabel('꽃받침너비')  
plt.ylabel('밀도 누적') 
plt.show()
```


    
![png](output_14_0.png)
    



```python
plt.hist(iris['꽃받침너비'], bins=10, orientation='horizontal')  # orientation 방향 
plt.ylabel('꽃받침너비')  
plt.xlabel('개수') 
plt.show()
```


    
![png](output_15_0.png)
    



```python
plt.hist(iris[iris.종=='setosa']['꽃잎길이'],
        color='blue', label='setosa', alpha=0.5)
plt.hist(iris[iris.종=='versicolor']['꽃잎길이'],
        color='red', label='versicolor', alpha=0.5)
plt.hist(iris[iris.종=='virginica']['꽃잎길이'],
        color='green', label='virginica', alpha=0.5)

plt.legend(title='종')
plt.xlabel('꽃잎길이')  
plt.ylabel('개수') 
plt.show()
```


    
![png](output_16_0.png)
    



```python
iris_g = iris.꽃잎길이.groupby([iris.종])
iris_g.describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>종</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>50.0</td>
      <td>1.462</td>
      <td>0.173664</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>1.50</td>
      <td>1.575</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>50.0</td>
      <td>4.260</td>
      <td>0.469911</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.35</td>
      <td>4.600</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>50.0</td>
      <td>5.552</td>
      <td>0.551895</td>
      <td>4.5</td>
      <td>5.1</td>
      <td>5.55</td>
      <td>5.875</td>
      <td>6.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris[iris.종=='setosa'].꽃잎길이.max()
```




    1.9




```python
iris.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   꽃받침길이   150 non-null    float64
     1   꽃받침너비   150 non-null    float64
     2   꽃잎길이    150 non-null    float64
     3   꽃잎너비    150 non-null    float64
     4   종       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    


```python
iris.describe()['꽃잎길이']
```




    count    150.000000
    mean       3.758000
    std        1.765298
    min        1.000000
    25%        1.600000
    50%        4.350000
    75%        5.100000
    max        6.900000
    Name: 꽃잎길이, dtype: float64




```python
attr_dict = {'꽃받침길이': 'blue',
            '꽃받침너비': 'red',
            '꽃잎길이': 'green',
            '꽃잎너비': 'yellow'}

for attr in attr_dict:
    plt.hist(iris[attr], color=attr_dict[attr], label=attr, alpha=0.3)  

plt.legend(title='속성', prop={'size':12}) #  prop속성은 범례의 특징을 작성해주는 속성  
plt.ylabel('개수') 

plt.show()
```


    
![png](output_21_0.png)
    



```python
f, a = plt.subplots(2, 2, figsize=(16, 9))

plt.subplot(2, 2, 1)
plt.hist(iris['꽃받침길이'], color='blue', label='꽃받침길이')
plt.legend(title='', prop={'size':12})
plt.ylabel('개수') 
plt.subplot(2, 2, 2)
plt.hist(iris['꽃받침너비'], color='red', label='꽃받침너비')
plt.legend(title='', prop={'size':13})
plt.ylabel('개수') 
plt.subplot(2, 2, 3)
plt.hist(iris['꽃잎길이'], color='green', label='꽃잎길이')
plt.legend(title='', prop={'size':14})
plt.ylabel('개수') 
plt.subplot(2, 2, 4)
plt.hist(iris['꽃잎너비'], color='yellow', label='꽃잎너비')
plt.legend(title='', prop={'size':15})
plt.ylabel('개수') 
plt.show()
```


    
![png](output_22_0.png)
    


# seaborn 라이브러리를 이용하여 히스토그램


```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정

# iris = pd.read_csv('./iris.csv') # 로컬 내려 받아 실행 할 경우
iris = pd.read_csv(iris_path)      # 온라인으로 실행 할 경우

iris.columns = ['꽃받침길이','꽃받침너비','꽃잎길이','꽃잎너비','종']
```


```python
sns.distplot(iris['꽃받침너비'], bins=10)  # 추세선까지 보여줌
plt.show()
sns.histplot(iris['꽃받침너비'], bins=10)
plt.show()
```

    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
![png](output_25_1.png)
    



    
![png](output_25_2.png)
    



```python
a = sns.distplot(iris['꽃받침너비'],  
                hist=True,                    # 히스토그램을 표시할 여부
                kde=True,                     # 추세선(kernel density curve)를 표시할 여부
                bins=10,                       # 막대의 개수
                color='blue',                  # 그래프의 색상
                hist_kws={'edgecolor': 'red'}, # 히스토그램의 옵션을 설정합니다. 여기서는 히스토그램의 테두리를 빨간색으로 설정
                kde_kws={'linewidth': 2})      # 추세선의 옵션을 설정합니다. 여기서는 선 두께를 2로 설정

a.set_title('꽃받침너비 histogram 히스토그램') # 그래프 제목 설정
a.set_xlabel('꽃받침너비')                    # x축 제목
a.set_ylabel('밀도')                        # y축 제목
plt.show()
```

    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
![png](output_26_1.png)
    



```python
sns.distplot(iris[iris.종=='setosa']['꽃잎길이'],
            color='blue', label='setosa')
sns.distplot(iris[iris.종=='versicolor']['꽃잎길이'],
            color='red', label='versicolor')
sns.distplot(iris[iris.종=='virginica']['꽃잎길이'],
            color='green', label='virginica')

plt.legend(title='종') 
plt.ylabel('밀도')                        # y축 제목

plt.show()
```

    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
![png](output_27_1.png)
    



```python
iris_g = iris.groupby('종').꽃잎너비.sum()
iris_g
```




    종
    setosa         12.3
    versicolor     66.3
    virginica     101.3
    Name: 꽃잎너비, dtype: float64



# 막대그래프


```python
import pandas as pd
# tips = pd.read_csv('./tips.csv')  # 로컬 내려 받아 실행 할 경우
tips = pd.read_csv(tips_path)      # 온라인으로 실행 할 경우

print(tips.shape)
print(tips.head())
print(tips.tail())
print(tips.describe()) 
tips.columns = ['총금액','팁','성별','흡연여부','요일','시간','규모']
print(tips.describe()) 
```

    (244, 7)
       total_bill   tip     sex smoker  day    time  size
    0       16.99  1.01  Female     No  Sun  Dinner     2
    1       10.34  1.66    Male     No  Sun  Dinner     3
    2       21.01  3.50    Male     No  Sun  Dinner     3
    3       23.68  3.31    Male     No  Sun  Dinner     2
    4       24.59  3.61  Female     No  Sun  Dinner     4
         total_bill   tip     sex smoker   day    time  size
    239       29.03  5.92    Male     No   Sat  Dinner     3
    240       27.18  2.00  Female    Yes   Sat  Dinner     2
    241       22.67  2.00    Male    Yes   Sat  Dinner     2
    242       17.82  1.75    Male     No   Sat  Dinner     2
    243       18.78  3.00  Female     No  Thur  Dinner     2
           total_bill         tip        size
    count  244.000000  244.000000  244.000000
    mean    19.785943    2.998279    2.569672
    std      8.902412    1.383638    0.951100
    min      3.070000    1.000000    1.000000
    25%     13.347500    2.000000    2.000000
    50%     17.795000    2.900000    2.000000
    75%     24.127500    3.562500    3.000000
    max     50.810000   10.000000    6.000000
                  총금액           팁          규모
    count  244.000000  244.000000  244.000000
    mean    19.785943    2.998279    2.569672
    std      8.902412    1.383638    0.951100
    min      3.070000    1.000000    1.000000
    25%     13.347500    2.000000    2.000000
    50%     17.795000    2.900000    2.000000
    75%     24.127500    3.562500    3.000000
    max     50.810000   10.000000    6.000000
    


```python
tips.columns
```




    Index(['총금액', '팁', '성별', '흡연여부', '요일', '시간', '규모'], dtype='object')




```python
tips_day = tips.groupby('요일').팁.sum()
tips_day
```




    요일
    Fri      51.96
    Sat     260.40
    Sun     247.39
    Thur    171.83
    Name: 팁, dtype: float64




```python
import numpy as np
label = ['Thur', 'Fri', 'Sat', 'Sun']
index = np.arange(len(label))

plt.bar(index, tips_day, 
        color='red', # 색
        alpha=0.3,   # 투명도
        width=0.5,   # 너비
        align='edge')

plt.title('요일별 팁 합계', fontsize=20)
plt.xlabel('요일', fontsize=15)
plt.ylabel('팁 합계', fontsize=15)
plt.xticks(index, label, fontsize=12,
           rotation=45)  # label을 회전 
plt.show()
```


    
![png](output_33_0.png)
    



```python
plt.barh(index, tips_day)
plt.title('요일별 팁 합계 bar graph')
plt.ylabel('요일')
plt.xlabel('팁 합계')
plt.yticks(index, label, rotation=0)

plt.show()
```


    
![png](output_34_0.png)
    



```python
label = ['Thur', 'Fri', 'Sat', 'Sun']
index = np.arange(len(label))

male_tip = tips[tips['성별']=='Male'].groupby('요일').팁.sum()
female_tip = tips[tips['성별']=='Female'].groupby('요일').팁.sum()

p1 = plt.bar(index, male_tip, color='blue', alpha=0.5)
p2 = plt.bar(index, female_tip, color='red', alpha=0.5,
            bottom=male_tip)  # bottom 속성을 이용하여 두 개의 그래프를 쌓을 수 있음

plt.title('성별 bar chart')
plt.ylabel('팁 합계')
plt.xlabel('요일')
plt.xticks(index, label)
plt.legend((p1[0], p2[0]), ('Male', 'Female'))
plt.show()
# p2를 보시면 bottom 속성이 추가된 것을 알 수 있는데, p2 그래프 아래에 p1 그래프를 넣겠다는 뜻
```


    
![png](output_35_0.png)
    



```python
p1 = plt.bar(index, male_tip, width=0.3, color='blue',
            alpha=0.5, label='Male')
p2 = plt.bar(index+0.3, female_tip, width=0.3, color='red',  # p2의 index 속성에 너비 값을 더하여 p2 그래프가 나타나는 위치를 바꿔준 것입니다
             alpha=0.5, label='Female')

plt.title('bar chart male/female')
plt.ylabel('tips sum')
plt.xlabel('day')
plt.xticks(index, label)
plt.legend((p1[0], p2[0]), ('Male', 'Female'))
plt.show()
```


    
![png](output_36_0.png)
    


## seaborn 라이브러리를 이용하여 bar 차트


```python
import seaborn as sns
tips_day = tips.groupby('요일').팁.sum()
label = ['Thur', 'Fri', 'Sat', 'Sun']

sns.barplot(x=label, y=tips_day, color='blue')   # barplot 함수를 사용하게 되며, x, y 속성에 값을 넣게 됩
plt.title('tips bar chart')

plt.show() 
```


    
![png](output_38_0.png)
    



```python
tips_day_sex = pd.DataFrame(tips.groupby(['요일', '성별']).팁.sum())
tips_day_sex
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
      <th>팁</th>
    </tr>
    <tr>
      <th>요일</th>
      <th>성별</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>Female</th>
      <td>25.03</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>26.93</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>Female</th>
      <td>78.45</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>181.95</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>Female</th>
      <td>60.61</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>186.78</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>Female</th>
      <td>82.42</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>89.41</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips_day_sex = tips_day_sex.reset_index()
tips_day_sex
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
      <th>요일</th>
      <th>성별</th>
      <th>팁</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>Female</td>
      <td>25.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Male</td>
      <td>26.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>Female</td>
      <td>78.45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Male</td>
      <td>181.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>Female</td>
      <td>60.61</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Male</td>
      <td>186.78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>Female</td>
      <td>82.42</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Male</td>
      <td>89.41</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips_day_sex = tips_day_sex.sort_values(by=['요일', '성별'], ascending=False)  # 누적 값을 만들어주기 위해 ascending 속성을 이용하여 정렬을 뒤집습니다
tips_day_sex
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
      <th>요일</th>
      <th>성별</th>
      <th>팁</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Male</td>
      <td>89.41</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>Female</td>
      <td>82.42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Male</td>
      <td>186.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>Female</td>
      <td>60.61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Male</td>
      <td>181.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>Female</td>
      <td>78.45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Male</td>
      <td>26.93</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>Female</td>
      <td>25.03</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips_day_sex['팁누적합계'] = tips_day_sex.groupby(['요일'])['팁'].cumsum(axis=0)
# cumsum 함수를 이용하여 누적 값을 구해줍니다
tips_day_sex 
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
      <th>요일</th>
      <th>성별</th>
      <th>팁</th>
      <th>팁누적합계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Male</td>
      <td>89.41</td>
      <td>89.41</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>Female</td>
      <td>82.42</td>
      <td>171.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Male</td>
      <td>186.78</td>
      <td>186.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>Female</td>
      <td>60.61</td>
      <td>247.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Male</td>
      <td>181.95</td>
      <td>181.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>Female</td>
      <td>78.45</td>
      <td>260.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Male</td>
      <td>26.93</td>
      <td>26.93</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>Female</td>
      <td>25.03</td>
      <td>51.96</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips_day_sex = tips_day_sex.sort_values(by=['요일', '성별'], ascending=True)
# 누적 값을 구하기 위해 뒤집었던 데이터프레임을 다시 뒤집습니다
tips_day_sex 
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
      <th>요일</th>
      <th>성별</th>
      <th>팁</th>
      <th>팁누적합계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>Female</td>
      <td>25.03</td>
      <td>51.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Male</td>
      <td>26.93</td>
      <td>26.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>Female</td>
      <td>78.45</td>
      <td>260.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Male</td>
      <td>181.95</td>
      <td>181.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>Female</td>
      <td>60.61</td>
      <td>247.39</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Male</td>
      <td>186.78</td>
      <td>186.78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>Female</td>
      <td>82.42</td>
      <td>171.83</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Male</td>
      <td>89.41</td>
      <td>89.41</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(x='요일', y='팁누적합계', hue='성별', data=tips_day_sex,
           dodge=False)

plt.title('Cumsum of tips')
plt.show()
# seaborn 라이브러리에서는 막대 그래프를 그릴 때 barplot 함수를 이용합니다
# hue 속성은 범례의 제목을 나타내며, dodge 속성은 그래프를 겹칠지 여부
```


    
![png](output_44_0.png)
    



```python
sns.barplot(x='요일', y='팁', hue='성별', data=tips_day_sex)
# 양 옆으로 놓인 그래프를 그릴 때에는 y 값을 그냥 여러 개 있는 것으로 선택
plt.title('Tips')
plt.show()
```


    
![png](output_45_0.png)
    



```python
tips_day = pd.DataFrame(tips.groupby('요일').팁.sum())
print(tips_day,'\n')
tips_day = tips_day.reset_index()
print(tips_day,'\n')
tips_day.plot.bar(x='요일', y='팁', rot=45)
plt.show()
# rot라는 속성이 있는데 이 속성은 그래프 x축의 제목들을 돌려주는 속성입니다
# 기본 값으로 90도가 돌아가 있으며 정방향으로 보려면 rot=0
```

               팁
    요일          
    Fri    51.96
    Sat   260.40
    Sun   247.39
    Thur  171.83 
    
         요일       팁
    0   Fri   51.96
    1   Sat  260.40
    2   Sun  247.39
    3  Thur  171.83 
    
    


    
![png](output_46_1.png)
    


# 파이 차트


```python
이름 = ['A 팀', 'B 팀', 'C 팀']
점수 = [97, 30, 67]
색상 = ['yellowgreen', 'lightcoral', 'skyblue']
중심과의거리 = (0.1, 0, 0)
표기포맷 = '%1.2f%%'

plt.pie(점수, explode=중심과의거리,
       labels=이름, colors=색상,
       autopct= 표기포맷, shadow=True, # shadow는 그림자 여부
       startangle=90) # startangle은 시작 각도

plt.axis('equal')
plt.title('pie chart of score')
plt.show() 
```


    
![png](output_48_0.png)
    



```python
names = ['Team A', 'Team B', 'Team C']
scores = [97, 30, 67]


c1, c2, c3 = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]  # plt.cm은 색상 팔레트를 의미
width_num = 0.4

flg, ax = plt.subplots()
ax.axis('equal')
pie_outer, _ = ax.pie(scores, radius=1.2, labels=names,
                      labeldistance=0.3,  # 중심으로부터 label이 생성될 위치
                      colors=[c1(0.5), c2(0.5), c3(0.5)])
plt.show()
```


    
![png](output_49_0.png)
    



```python
fig, ax = plt.subplots()
ax.axis('equal')
pie_outer, _ = ax.pie(scores, radius=1.2, labels=names,
                     labeldistance=0.8, colors=[c1(0.5), c2(0.5), c3(0.5)])
plt.setp(pie_outer, width=0.4, edgecolor='white')
# setp 함수를 이용하여 너비와 가장자리의 색상을 지정
plt.show() 
```


    
![png](output_50_0.png)
    



```python
subnames = ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2']
subsize = [37, 29, 31, 19, 11, 58, 9]

fig, ax = plt.subplots()
ax.axis('equal')
pie_outer, _ = ax.pie(scores, radius=1.2, labels=names,
                     labeldistance=0.8, colors=[c1(0.5), c2(0.5), c3(0.5)])
plt.setp(pie_outer, width=0.4, edgecolor='white')

pie_inner, plt_labels, junk = ax.pie(subsize, radius=0.8, labels=subnames,
                                    labeldistance=0.75, autopct='%1.1f%%',
                                    colors=[c1(0.3), c1(0.2), c1(0.1),
                                            c2(0.3), c2(0.2), 
                                            c3(0.3), c3(0.2)])
plt.show()
# 똑같이 pie 함수를 사용하며, 그냥 큰 도넛 파이 차트 내부에 작은 파이 차트 하나를 추가
```


    
![png](output_51_0.png)
    


# 산점도
산점도 그래프는 두 개의 연속형 변수에 대한 관계를 파악하는데 사용하는 그래프


```python
# iris = sns.load_dataset('iris')
import pandas as pd
# iris = pd.read_csv('./iris.csv')   # 로컬 내려 받아 실행 할 경우
iris = pd.read_csv(iris_path)        # 온라인으로 실행 할 경우

iris.columns = ['꽃받침길이','꽃받침너비','꽃잎길이','꽃잎너비','종']

plt.plot('꽃잎길이', '꽃잎너비', data=iris,
         linestyle='none', marker='o', markersize=10,
         color='blue', alpha=0.5)

plt.title('iris scatter plot')
plt.xlabel('꽃잎길이')
plt.ylabel('꽃잎너비')
plt.show()
# 첫 두 개 매개변수는 x와 y 변수, data 속성은 데이터셋, linestyle은 후에 나올 선의 스타일,
# marker는 점의 속성, markersize는 점의 크기, color는 점의 색상, alpha는 점의 투명도
```


    
![png](output_53_0.png)
    



```python
import matplotlib.patches as patches   # patches 함수를 이용하여 도형을 추가
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot('꽃잎길이', '꽃잎너비', data=iris,
        linestyle='none', marker='o')


# 직사각형의 패치를 생성
# 각각의 속성들은 좌표, 크기, 투명도, 면색상, 테두리 색상, 테두리 두께, 테두리 모양, 각도
ax1.add_patch(patches.Rectangle((3, 1), 2, 1, alpha=0.2,
                               facecolor='blue', edgecolor='black',
                               linewidth=2, linestyle='solid',
                               angle=-10))

# 원형 패치를 생성
ax1.add_patch(patches.Circle((1.5, 0.25), 0.5, alpha=0.2,
                            facecolor='red', edgecolor='black',
                            linewidth=2, linestyle='--'))

# 선을 그랠때에는 patch가 아닌 plot을 이용
plt.plot([4, 6], [2.2, 1.1], color='green', lw=4, linestyle=':')

plt.xlabel('꽃잎길이')
plt.ylabel('꽃잎너비')
plt.show()
```


    
![png](output_54_0.png)
    



```python
groups = iris.groupby('종')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.꽃잎길이,
           group.꽃잎너비,
           marker='x',
           linestyle='',
           label=name)

plt.xlabel('꽃잎길이')
plt.ylabel('꽃잎너비')
ax.legend(loc='upper left')  # 범례의 위치
plt.title('iris scatter plot by species')
plt.show()
```


    
![png](output_55_0.png)
    



```python
groups.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">꽃받침길이</th>
      <th colspan="2" halign="left">꽃받침너비</th>
      <th>...</th>
      <th colspan="2" halign="left">꽃잎길이</th>
      <th colspan="8" halign="left">꽃잎너비</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>종</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>50.0</td>
      <td>5.006</td>
      <td>0.352490</td>
      <td>4.3</td>
      <td>4.800</td>
      <td>5.0</td>
      <td>5.2</td>
      <td>5.8</td>
      <td>50.0</td>
      <td>3.428</td>
      <td>...</td>
      <td>1.575</td>
      <td>1.9</td>
      <td>50.0</td>
      <td>0.246</td>
      <td>0.105386</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>50.0</td>
      <td>5.936</td>
      <td>0.516171</td>
      <td>4.9</td>
      <td>5.600</td>
      <td>5.9</td>
      <td>6.3</td>
      <td>7.0</td>
      <td>50.0</td>
      <td>2.770</td>
      <td>...</td>
      <td>4.600</td>
      <td>5.1</td>
      <td>50.0</td>
      <td>1.326</td>
      <td>0.197753</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.3</td>
      <td>1.5</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>50.0</td>
      <td>6.588</td>
      <td>0.635880</td>
      <td>4.9</td>
      <td>6.225</td>
      <td>6.5</td>
      <td>6.9</td>
      <td>7.9</td>
      <td>50.0</td>
      <td>2.974</td>
      <td>...</td>
      <td>5.875</td>
      <td>6.9</td>
      <td>50.0</td>
      <td>2.026</td>
      <td>0.274650</td>
      <td>1.4</td>
      <td>1.8</td>
      <td>2.0</td>
      <td>2.3</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 32 columns</p>
</div>




```python
# seaborn 라이브러리에서는 regplot 함수를 이용하여 scatter plot을 그립니다
sns.regplot(x=iris['꽃잎길이'],
           y=iris['꽃잎너비'],
           fit_reg=True) # fit_reg 속성은 추세선을 표시할지 여부

plt.title('seaborn scatter plot')
plt.show()
```


    
![png](output_57_0.png)
    



```python
# seaborn 라이브러리에서 순수하게(추세선을 제외하고) 산점도만 그리려면 scatterplot 함수를 사용합니다
# 속성 값이 약간 달라지게 되는데, data에 데이터셋을 넣고, x와 y에 넣을 데이터의 이름이 들어가게 됩니다
sns.scatterplot(x='꽃잎길이',
               y='꽃잎너비',
               alpha=0.3,
               data=iris,
               color='red')
plt.show() 
```


    
![png](output_58_0.png)
    



```python
sns.scatterplot(x='꽃잎길이',
               y='꽃잎너비',
               hue='종',       # 범례
               style='종',     # style 또한 종 별로 다르게
               s=100, data=iris)
plt.show() 
```


    
![png](output_59_0.png)
    



```python
sns.pairplot(iris)
plt.show() 
# pairplot 함수를 이용하면 모든 변수끼리의 관계를 보여주게 됩니다
```


    
![png](output_60_0.png)
    



```python
sns.pairplot(iris, 
             # diag_kind 속성은 대각선에 놓이는 그래프들, 즉 두개가 같은 그래프들을 어떤 그래프로 표현할지를 나타냄
             diag_kind='kde',   # kde는 밀도 그래프라고 생각하면됨
             hue='종',          # 종 별로 범례 색상을 나누게 됨
             # palette에는 bright, pastel, deep, muted, colorblind, dark 등의 종류가 있음
             palette='bright')  # palette는 색상들의 전체적인 속성
plt.show()
```


    
![png](output_61_0.png)
    


# pandas 라이브러리를 이용하여 산점도 


```python
# pandas 라이브러리에서 산점도
iris.plot.scatter(x='꽃잎길이',
                 y='꽃잎너비',
                 s=50,     # 점의 크기
                 c='blue', # 색상
                 alpha=0.5)
plt.show()
```


    
![png](output_63_0.png)
    



```python
iris['color'] = np.where(iris.종=='setosa', 'red',               # np.where 함수는 논리형에 만족하는 값
                        np.where(iris.종=='versicolor', 'green', 
                                 'blue'))
iris.plot(kind='scatter', x='꽃잎길이', y='꽃잎너비',
         s=50, c=iris['color'])
plt.show() 
```


    
![png](output_64_0.png)
    



```python
# pandas도 산점도 행렬
# pandas.plotting에 있는 scatter_matrix 함수를 이용
from pandas.plotting import scatter_matrix
scatter_matrix(iris, alpha=0.5, figsize=(8,8), diagonal='kde')
plt.show()
```


    
![png](output_65_0.png)
    



```python
shapes = list('.,ov^><1234sp*hHDd|_x')
num = 0
for x in range(1, 4):
    for y in range(1, 8):
        num += 1
        plt.plot(x, y,
                marker=shapes[num-1],
                markerfacecolor='green',
                markersize=10,
                markeredgecolor='black')
        plt.text(x+0.1, y,
                shapes[num-1],
                horizontalalignment='left',
                size='medium',
                color='black',
                weight='semibold')
        
plt.title('marker')
plt.xticks([1, 2, 3, 4])
plt.show() 
# 맨 윗 줄의 shapes 리스트를 보시면 해당 리스트는 설정할 수 있는 모양들의 리스트입니다
# 해당 문자에 따른 모양은 아래 그래프를 통해 그려져있습니다
```


    
![png](output_66_0.png)
    


# 꺾은 선 그래프


```python
index = pd.date_range('20210101', periods=100, freq='m', name='Date')
data = np.random.randint(1, 11, (100, 4)).cumsum(axis=0)

wide_df = pd.DataFrame(data, index, list('abcd'))
print(wide_df.shape)
print(wide_df.head())
print(wide_df.tail())
print(wide_df.describe())
```

    (100, 4)
                 a   b   c   d
    Date                      
    2021-01-31  10   7  10   3
    2021-02-28  13  17  17   9
    2021-03-31  17  21  19  18
    2021-04-30  19  29  22  22
    2021-05-31  27  35  23  28
                  a    b    c    d
    Date                          
    2028-12-31  500  529  534  522
    2029-01-31  501  530  539  529
    2029-02-28  509  537  548  531
    2029-03-31  510  547  556  534
    2029-04-30  520  550  560  544
                    a           b           c           d
    count  100.000000  100.000000  100.000000  100.000000
    mean   260.980000  273.440000  282.040000  268.390000
    std    150.594913  152.770464  168.586693  152.045972
    min     10.000000    7.000000   10.000000    3.000000
    25%    131.500000  149.750000  131.500000  143.250000
    50%    263.000000  283.000000  283.000000  269.000000
    75%    399.000000  398.000000  428.750000  389.750000
    max    520.000000  550.000000  560.000000  544.000000
    


```python
long_df = wide_df.stack()
print(long_df.shape)
print(long_df.head(8), '\n------------------\n')

long_df = pd.DataFrame(long_df).reset_index()
print(long_df.head())

long_df.columns = ['Date', 'Group', 'CumVal']
print(long_df.shape)
print(long_df.head()) 

# 위에서 나온 데이터를 길게 늘어뜨려 가공하였습니다
```

    (400,)
    Date         
    2021-01-31  a    10
                b     7
                c    10
                d     3
    2021-02-28  a    13
                b    17
                c    17
                d     9
    dtype: int32 
    ------------------
    
            Date level_1   0
    0 2021-01-31       a  10
    1 2021-01-31       b   7
    2 2021-01-31       c  10
    3 2021-01-31       d   3
    4 2021-02-28       a  13
    (400, 3)
            Date Group  CumVal
    0 2021-01-31     a      10
    1 2021-01-31     b       7
    2 2021-01-31     c      10
    3 2021-01-31     d       3
    4 2021-02-28     a      13
    


```python
long_df['Size'] = np.where(long_df['Group']=='a', 1,
                          np.where(long_df['Group']=='b', 2,
                                  np.where(long_df['Group']=='c', 3,
                                          4)))
long_df.head(n=12)
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
      <th>Date</th>
      <th>Group</th>
      <th>CumVal</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-31</td>
      <td>a</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-31</td>
      <td>b</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-31</td>
      <td>c</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-31</td>
      <td>d</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-28</td>
      <td>a</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-02-28</td>
      <td>b</td>
      <td>17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-02-28</td>
      <td>c</td>
      <td>17</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021-02-28</td>
      <td>d</td>
      <td>9</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2021-03-31</td>
      <td>a</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2021-03-31</td>
      <td>b</td>
      <td>21</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2021-03-31</td>
      <td>c</td>
      <td>19</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2021-03-31</td>
      <td>d</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(wide_df.index, wide_df.a, marker='s', color='b')
plt.plot(wide_df.index, wide_df.b, marker='o', color='r')
plt.plot(wide_df.index, wide_df.c, marker='*', color='g')
plt.plot(wide_df.index, wide_df.d, marker='+', color='y')

plt.legend(wide_df.columns)
plt.show()
```


    
![png](output_71_0.png)
    



```python
line = ['--', '-', ':', '-.']
width = range(1, 5)

for i in range(0, 4):
    plt.plot(wide_df.index, wide_df.iloc[:, i],
            linestyle=line[i], linewidth=width[i])

plt.legend(wide_df.columns)
plt.show()
```


    
![png](output_72_0.png)
    



```python
group = list('abcd')
linewidths = range(1, 5)

for groups, size in zip(group, linewidths):
    long_df_ = long_df[long_df['Group'] == groups]
    plt.plot(long_df_.Date, long_df_.CumVal, linewidth=size)
    
plt.legend(group)
plt.show()
```


    
![png](output_73_0.png)
    


## seaborn 라이브러리 꺽은선


```python
ax = sns.lineplot(data=wide_df)
plt.legend(loc='best') #  legend 함수 안에 loc 속성이 best로 되어있는데, 가장 최상의 위치에 범례를 배치한다는 뜻
plt.show() 
```


    
![png](output_75_0.png)
    



```python
ax = sns.lineplot(x='Date', y='CumVal', style='Size',  # style을 이용하여 데이터 행들을 분류 
                 data=long_df)
plt.legend(loc='best')
plt.show() 
```


    
![png](output_76_0.png)
    



```python
ax = sns.lineplot(x='Date', y='CumVal', style='Size',
                 hue='Group', data=long_df)
plt.legend(loc='best')
plt.show()

# 위의 그래프에서 hue 속성을 추가하여 group 별로 색상을 입혀보았습니다
# 당연히 우리 데이터에서는 group끼리 size가 동일하기 때문에 한 선에 한 색상이 부여되어있습니다
```


    
![png](output_77_0.png)
    



```python
wide_df.plot.line()   # pandas 라이브러리는 plot.line() 함수를 이용 
plt.legend(loc='best')
plt.show() 
```


    
![png](output_78_0.png)
    


# 모자이크 그래프


```python
from statsmodels.graphics.mosaicplot import mosaic 
# titanic = pd.read_csv('./train.csv')  # 로컬 내려 받아 실행 할 경우
titanic = pd.read_csv(train_path)       # 온라인으로 실행 할 경우
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 11 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   survived  891 non-null    int64  
     1   pclass    891 non-null    int64  
     2   name      891 non-null    object 
     3   sex       891 non-null    object 
     4   age       714 non-null    float64
     5   sibsp     891 non-null    int64  
     6   parch     891 non-null    int64  
     7   ticket    891 non-null    object 
     8   fare      891 non-null    float64
     9   cabin     204 non-null    object 
     10  embarked  889 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 76.7+ KB
    


```python
titanic = titanic[['survived', 'pclass', 'sex']]
titanic["생존여부"] = titanic.survived.map({0: "사망", 1: "생존"})
titanic["등급"] = titanic.pclass.map({1: "1등석", 2: "2등석", 3: "3등석"})
titanic["성별"] = titanic.sex.map({'male': '남자', 'female': "여자"})

print(titanic.head())
print(titanic.describe())
```

       survived  pclass     sex 생존여부   등급  성별
    0         0       3    male   사망  3등석  남자
    1         1       1  female   생존  1등석  여자
    2         1       3  female   생존  3등석  여자
    3         1       1  female   생존  1등석  여자
    4         0       3    male   사망  3등석  남자
             survived      pclass
    count  891.000000  891.000000
    mean     0.383838    2.308642
    std      0.486592    0.836071
    min      0.000000    1.000000
    25%      0.000000    2.000000
    50%      0.000000    3.000000
    75%      1.000000    3.000000
    max      1.000000    3.000000
    


```python
mosaic(titanic.sort_values('등급'),
      ['생존여부', '등급'],
      title='titanic mosaic')

plt.show() 
```


    
![png](output_82_0.png)
    



```python
mosaic(titanic, ['생존여부', '성별'])
plt.title('titanic mosaic')
plt.show() 
```


    
![png](output_83_0.png)
    



```python
mosaic(titanic.sort_values('등급'),
      ['등급', '생존여부', '성별'])

plt.show() 
```


    
![png](output_84_0.png)
    



```python
mosaic(titanic.sort_values('등급'),
      ['등급', '생존여부', '성별'],
      gap = 0.03)

plt.show() 
```


    
![png](output_85_0.png)
    


# 히트 맵


```python
import pandas as pd
# flights = pd.read_csv('./flights.csv') # 로컬 내려 받아 실행 할 경우
flights = pd.read_csv(flights_path)      # 온라인으로 실행 할 경우

print(flights.shape)
print(flights.head())

df = flights.pivot('month', 'year', 'passengers')
print(df.head()) 
```

    (144, 3)
       year     month  passengers
    0  1949   January         112
    1  1949  February         118
    2  1949     March         132
    3  1949     April         129
    4  1949       May         121
    year      1949  1950  1951  1952  1953  1954  1955  1956  1957  1958  1959  \
    month                                                                        
    April      129   135   163   181   235   227   269   313   348   348   396   
    August     148   170   199   242   272   293   347   405   467   505   559   
    December   118   140   166   194   201   229   278   306   336   337   405   
    February   118   126   150   180   196   188   233   277   301   318   342   
    January    112   115   145   171   196   204   242   284   315   340   360   
    
    year      1960  
    month           
    April      461  
    August     606  
    December   432  
    February   391  
    January    417  
    


```python
plt.pcolor(df)

plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.title('heat map flights')
plt.colorbar()
plt.show() 
"""heat map은 pcolor 함수를 이용하여 생성합니다

xticks, yticks는 예전에도 했다시피 x축과 y축의 이름을 지정합니다

colorbar는 그래프 우측에 보이는 색상 바를 표시해주는지 여부를 결정합니다
"""
```


    
![png](output_88_0.png)
    





    'heat map은 pcolor 함수를 이용하여 생성합니다\n\nxticks, yticks는 예전에도 했다시피 x축과 y축의 이름을 지정합니다\n\ncolorbar는 그래프 우측에 보이는 색상 바를 표시해주는지 여부를 결정합니다\n'




```python
ax = sns.heatmap(df)
plt.show() 
```


    
![png](output_89_0.png)
    



```python
sns.heatmap(df, annot=True, fmt='d')
plt.show() 
```


    
![png](output_90_0.png)
    


annot 속성은 그래프에 숫자 값을 표현할지 여부를 선택합니다

fmt 속성은 해당 숫자 값을 어떻게 표현할지 포맷을 결정합니다. 'd' 이므로 정수형으로 표현됩니다 


```python
sns.heatmap(df, cmap='RdYlGn_r')
plt.show() 
# cmap 속성은 색상 팔레트를 지정해줍니다
# 여기서는 Red Yellow Green 팔레트를 지정
```


    
![png](output_92_0.png)
    



```python
sns.heatmap(df, center=df.loc['April', 1960], cmap='RdYlGn_r')
plt.show() 
# center 속성은 colormap의 중심을 해당 값으로 하겠다는 뜻입니다
# 즉, 여기서는 1960년 1월의 값을 컬러맵의 중심으로 하여 heatmap이 생성됩니다
# 자세히 보시면 위와는 그래프 색상이 약간 달라졌음을 확인하실 수 있습니다
```


    
![png](output_93_0.png)
    



```python
df.style.background_gradient(cmap='summer')
# pandas 라이브러리의 경우에는 데이터프레임의 배경에 색상을 입히는 것으로 heat map을 생성할 수 있습니다
```




<style  type="text/css" >
#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col0{
            background-color:  #91c866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col1{
            background-color:  #60b066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col2{
            background-color:  #55aa66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col3{
            background-color:  #249266;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col4,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col10{
            background-color:  #99cc66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col5,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col6,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col0{
            background-color:  #57ab66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col6,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col8{
            background-color:  #46a266;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col7,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col10{
            background-color:  #4ba566;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col8,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col6,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col3{
            background-color:  #48a366;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col9{
            background-color:  #319866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col10,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col7,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col7{
            background-color:  #3f9f66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col11,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col11{
            background-color:  #4ea666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col0,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col2,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col3,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col4,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col8,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col9,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col10,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col0,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col2,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col5,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col6,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col7,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col11{
            background-color:  #ffff66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col5{
            background-color:  #ebf566;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col6{
            background-color:  #deee66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col7{
            background-color:  #f1f866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col11{
            background-color:  #eef666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col0,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col0{
            background-color:  #51a866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col1{
            background-color:  #76bb66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col2{
            background-color:  #63b166;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col3,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col7{
            background-color:  #52a866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col4{
            background-color:  #3a9c66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col5,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col10,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col5{
            background-color:  #5cae66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col8{
            background-color:  #359a66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col9,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col5{
            background-color:  #239166;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col10{
            background-color:  #4aa466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col11,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col0{
            background-color:  #2e9666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col1{
            background-color:  #369a66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col2,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col7,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col10{
            background-color:  #178b66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col3,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col11{
            background-color:  #209066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col4,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col4{
            background-color:  #2c9666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col5,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col6,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col8,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col10,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col2,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col3,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col0,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col4,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col7,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col9,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col11{
            background-color:  #008066;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col7,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col9{
            background-color:  #0a8466;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col11{
            background-color:  #018066;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col2{
            background-color:  #048266;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col6{
            background-color:  #118866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col8,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col10{
            background-color:  #158a66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col9{
            background-color:  #279366;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col11{
            background-color:  #1d8e66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col3{
            background-color:  #d4ea66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col4{
            background-color:  #e9f466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col8{
            background-color:  #fcfe66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col9{
            background-color:  #edf666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col10{
            background-color:  #f3f966;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col0{
            background-color:  #b4da66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col6,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col11{
            background-color:  #a0d066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col2,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col2{
            background-color:  #9cce66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col3{
            background-color:  #a9d466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col4{
            background-color:  #afd766;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col5{
            background-color:  #aad466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col7{
            background-color:  #b9dc66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col8,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col0{
            background-color:  #badc66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col9{
            background-color:  #a4d266;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col0{
            background-color:  #a2d066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col9{
            background-color:  #7bbd66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col3{
            background-color:  #4fa766;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col4{
            background-color:  #9bcd66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col5{
            background-color:  #69b466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col6{
            background-color:  #42a066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col8,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col7{
            background-color:  #54aa66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col9{
            background-color:  #44a266;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col0{
            background-color:  #62b066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col1{
            background-color:  #329866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col2{
            background-color:  #80c066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col3{
            background-color:  #2b9566;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col4{
            background-color:  #88c366;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col5{
            background-color:  #67b366;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col8{
            background-color:  #53a966;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col9{
            background-color:  #45a266;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col11{
            background-color:  #5aac66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col3{
            background-color:  #038166;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col5{
            background-color:  #219066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col6{
            background-color:  #078366;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col8{
            background-color:  #068266;
            color:  #f1f1f1;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col1,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col4{
            background-color:  #56ab66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col2,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col6{
            background-color:  #50a866;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col9{
            background-color:  #40a066;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col10{
            background-color:  #4ca666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col1{
            background-color:  #c9e466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col2{
            background-color:  #b8dc66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col3{
            background-color:  #89c466;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col4,#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col8{
            background-color:  #9ece66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col5{
            background-color:  #9fcf66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col6{
            background-color:  #9acc66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col7{
            background-color:  #97cb66;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col10{
            background-color:  #8ec666;
            color:  #000000;
        }#T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col11{
            background-color:  #82c066;
            color:  #000000;
        }</style><table id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879" ><thead>    <tr>        <th class="index_name level0" >year</th>        <th class="col_heading level0 col0" >1949</th>        <th class="col_heading level0 col1" >1950</th>        <th class="col_heading level0 col2" >1951</th>        <th class="col_heading level0 col3" >1952</th>        <th class="col_heading level0 col4" >1953</th>        <th class="col_heading level0 col5" >1954</th>        <th class="col_heading level0 col6" >1955</th>        <th class="col_heading level0 col7" >1956</th>        <th class="col_heading level0 col8" >1957</th>        <th class="col_heading level0 col9" >1958</th>        <th class="col_heading level0 col10" >1959</th>        <th class="col_heading level0 col11" >1960</th>    </tr>    <tr>        <th class="index_name level0" >month</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row0" class="row_heading level0 row0" >April</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col0" class="data row0 col0" >129</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col1" class="data row0 col1" >135</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col2" class="data row0 col2" >163</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col3" class="data row0 col3" >181</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col4" class="data row0 col4" >235</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col5" class="data row0 col5" >227</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col6" class="data row0 col6" >269</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col7" class="data row0 col7" >313</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col8" class="data row0 col8" >348</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col9" class="data row0 col9" >348</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col10" class="data row0 col10" >396</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row0_col11" class="data row0 col11" >461</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row1" class="row_heading level0 row1" >August</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col0" class="data row1 col0" >148</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col1" class="data row1 col1" >170</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col2" class="data row1 col2" >199</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col3" class="data row1 col3" >242</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col4" class="data row1 col4" >272</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col5" class="data row1 col5" >293</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col6" class="data row1 col6" >347</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col7" class="data row1 col7" >405</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col8" class="data row1 col8" >467</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col9" class="data row1 col9" >505</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col10" class="data row1 col10" >559</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row1_col11" class="data row1 col11" >606</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row2" class="row_heading level0 row2" >December</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col0" class="data row2 col0" >118</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col1" class="data row2 col1" >140</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col2" class="data row2 col2" >166</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col3" class="data row2 col3" >194</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col4" class="data row2 col4" >201</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col5" class="data row2 col5" >229</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col6" class="data row2 col6" >278</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col7" class="data row2 col7" >306</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col8" class="data row2 col8" >336</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col9" class="data row2 col9" >337</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col10" class="data row2 col10" >405</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row2_col11" class="data row2 col11" >432</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row3" class="row_heading level0 row3" >February</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col0" class="data row3 col0" >118</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col1" class="data row3 col1" >126</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col2" class="data row3 col2" >150</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col3" class="data row3 col3" >180</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col4" class="data row3 col4" >196</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col5" class="data row3 col5" >188</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col6" class="data row3 col6" >233</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col7" class="data row3 col7" >277</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col8" class="data row3 col8" >301</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col9" class="data row3 col9" >318</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col10" class="data row3 col10" >342</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row3_col11" class="data row3 col11" >391</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row4" class="row_heading level0 row4" >January</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col0" class="data row4 col0" >112</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col1" class="data row4 col1" >115</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col2" class="data row4 col2" >145</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col3" class="data row4 col3" >171</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col4" class="data row4 col4" >196</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col5" class="data row4 col5" >204</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col6" class="data row4 col6" >242</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col7" class="data row4 col7" >284</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col8" class="data row4 col8" >315</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col9" class="data row4 col9" >340</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col10" class="data row4 col10" >360</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row4_col11" class="data row4 col11" >417</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row5" class="row_heading level0 row5" >July</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col0" class="data row5 col0" >148</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col1" class="data row5 col1" >170</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col2" class="data row5 col2" >199</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col3" class="data row5 col3" >230</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col4" class="data row5 col4" >264</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col5" class="data row5 col5" >302</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col6" class="data row5 col6" >364</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col7" class="data row5 col7" >413</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col8" class="data row5 col8" >465</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col9" class="data row5 col9" >491</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col10" class="data row5 col10" >548</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row5_col11" class="data row5 col11" >622</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row6" class="row_heading level0 row6" >June</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col0" class="data row6 col0" >135</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col1" class="data row6 col1" >149</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col2" class="data row6 col2" >178</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col3" class="data row6 col3" >218</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col4" class="data row6 col4" >243</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col5" class="data row6 col5" >264</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col6" class="data row6 col6" >315</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col7" class="data row6 col7" >374</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col8" class="data row6 col8" >422</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col9" class="data row6 col9" >435</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col10" class="data row6 col10" >472</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row6_col11" class="data row6 col11" >535</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row7" class="row_heading level0 row7" >March</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col0" class="data row7 col0" >132</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col1" class="data row7 col1" >141</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col2" class="data row7 col2" >178</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col3" class="data row7 col3" >193</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col4" class="data row7 col4" >236</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col5" class="data row7 col5" >235</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col6" class="data row7 col6" >267</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col7" class="data row7 col7" >317</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col8" class="data row7 col8" >356</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col9" class="data row7 col9" >362</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col10" class="data row7 col10" >406</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row7_col11" class="data row7 col11" >419</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row8" class="row_heading level0 row8" >May</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col0" class="data row8 col0" >121</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col1" class="data row8 col1" >125</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col2" class="data row8 col2" >172</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col3" class="data row8 col3" >183</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col4" class="data row8 col4" >229</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col5" class="data row8 col5" >234</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col6" class="data row8 col6" >270</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col7" class="data row8 col7" >318</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col8" class="data row8 col8" >355</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col9" class="data row8 col9" >363</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col10" class="data row8 col10" >420</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row8_col11" class="data row8 col11" >472</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row9" class="row_heading level0 row9" >November</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col0" class="data row9 col0" >104</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col1" class="data row9 col1" >114</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col2" class="data row9 col2" >146</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col3" class="data row9 col3" >172</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col4" class="data row9 col4" >180</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col5" class="data row9 col5" >203</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col6" class="data row9 col6" >237</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col7" class="data row9 col7" >271</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col8" class="data row9 col8" >305</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col9" class="data row9 col9" >310</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col10" class="data row9 col10" >362</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row9_col11" class="data row9 col11" >390</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row10" class="row_heading level0 row10" >October</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col0" class="data row10 col0" >119</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col1" class="data row10 col1" >133</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col2" class="data row10 col2" >162</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col3" class="data row10 col3" >191</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col4" class="data row10 col4" >211</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col5" class="data row10 col5" >229</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col6" class="data row10 col6" >274</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col7" class="data row10 col7" >306</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col8" class="data row10 col8" >347</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col9" class="data row10 col9" >359</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col10" class="data row10 col10" >407</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row10_col11" class="data row10 col11" >461</td>
            </tr>
            <tr>
                        <th id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879level0_row11" class="row_heading level0 row11" >September</th>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col0" class="data row11 col0" >136</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col1" class="data row11 col1" >158</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col2" class="data row11 col2" >184</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col3" class="data row11 col3" >209</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col4" class="data row11 col4" >237</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col5" class="data row11 col5" >259</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col6" class="data row11 col6" >312</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col7" class="data row11 col7" >355</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col8" class="data row11 col8" >404</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col9" class="data row11 col9" >404</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col10" class="data row11 col10" >463</td>
                        <td id="T_ed4973c0_bd1c_11eb_ab77_be44b86f5879row11_col11" class="data row11 col11" >508</td>
            </tr>
    </tbody></table>



# 그래프를 사진으로 저장


```python
groups = iris.groupby('종')
fix, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.꽃잎길이,
           group.꽃잎너비,
           marker='x',
           linestyle='',
           label=name)

ax.legend(loc='upper left')
plt.title('iris scatter plot by species')

# 사진 저장할 때는 절대로 plt.show()를 해서는 안됩니다

plt.savefig('./plot3.png', dpi=200, facecolor='#c0ffee')  
# dpi 속성을 이용하면 사진의 dpi(해상도 비슷한 설정)
# facecolor 속성을 이용하면 배경색상을 지정
plt.savefig('./plot4.png', bbox_inches='tight')
# bbox_inches 속성은 저장할 그래프의 영역을 설정하는데,
# tight으로 했기 때문에 여백 없이 plot 만 저장 
```


    
![png](output_96_0.png)
    


# 박스 플롯


```python
# tips = pd.read_csv('./tips.csv')    # 로컬 내려 받아 실행 할 경우
tips = pd.read_csv(tips_path)         # 온라인으로 실행 할 경우

tips['총금액'] = tips.total_bill
tips['팁'] = tips.tip
tips['성별'] = tips.sex
tips['흡연여부'] = tips.smoker
tips['요일'] = tips.day
tips['시간'] = tips.time
tips['규모'] = tips.size

tips.groupby(['성별', '요일']).size()
```




    성별      요일  
    Female  Fri      9
            Sat     28
            Sun     18
            Thur    32
    Male    Fri     10
            Sat     59
            Sun     58
            Thur    30
    dtype: int64




```python
plt.boxplot(tips['팁'])
plt.show()
```


    
![png](output_99_0.png)
    



```python
plt.boxplot(tips['팁'], sym='bo')  # sym 속성은 값들에 대한 색깔과 모양을 설정
plt.title('팁 boxplot')            # 그래프의 제목
plt.xticks([1], ['팁'])            # xticks는 x축의 값들의 이름을 설정
plt.show()
```


    
![png](output_100_0.png)
    



```python
plt.boxplot(tips['팁'], 
            sym='rs',  # rs는 red 색상의 square 모양
            vert=0)    # ertical 속성으로 방향을 설정
plt.show()
```


    
![png](output_101_0.png)
    



```python
plt.boxplot(tips['팁'], 
            notch=1,            # 중간의 주황색 선 부분을 v자로 파줍
            sym='gx', vert=0)   # gx는 green 색상의 x 표시
plt.show()
```


    
![png](output_102_0.png)
    



```python
fig, ax = plt.subplots()   # 여러 그래프를 한 번에 표시
ax.boxplot([tips['총금액'], tips['팁']], sym='y*')  # y*는 yellow 색상의 * 모양
plt.xticks([1, 2], ['총금액', '팁'])
plt.show()
# boxplot도 subplot을 이용하여 여러 그래프를 한 번에 표시할 수 있습니다 
```


    
![png](output_103_0.png)
    



```python
sns.boxplot(tips['팁']) # seaborn 라이브러리
plt.show()
```

    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_104_1.png)
    



```python
sns.boxplot(x='요일', y='팁', data=tips)
plt.show() # seaborn 라이브러리를 이용하여 그래프를 그릴 경우, 다수의 그룹이 그려지게 된다 
```


    
![png](output_105_0.png)
    



```python
sns.boxplot(x='요일', y='팁', hue='성별', data=tips)  # hue를 이용하여 그룹을 짓고, 범례에 제목
plt.show()
```


    
![png](output_106_0.png)
    

