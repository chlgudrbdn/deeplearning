# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# pandas 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 불러올 때 각 컬럼에 해당하는 이름을 지정합니다.
df = pd.read_csv('../dataset/pima-indians-diabetes.csv',
               names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

# 처음 5줄을 봅니다.
print(df.head(5))

# 데이터의 전반적인 정보를 확인해 봅니다.
print(df.info())

# 각 정보별 특징을 좀더 자세히 출력합니다.
print(df.describe())

# 데이터 중 임신 정보와 클래스 만을 출력해 봅니다.
print(df[['plasma', 'class']]) # plasma는 공복 혈당 농도.

# 임신 회수당 발병 확률 구함. 두줄 끌어 온뒤 임신별로 그루핑하고 as_index로 새로 index만든 뒤 여기에 class값의 평균값 넣고 임신값 기준으로 오름차순 정렬.
print(df[['pregnant','class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))
print(df[['pregnant','class']].groupby(['pregnant']).mean())#라고 해도 별 차인 없는거 같다. 다만 데이터 구조가 SQL 스타일 처럼 되지 않고 df로 안 묶이는걸로 보인다. as_index=False라고 해야 할 듯.
# 데이터 간의 상관관계를 그래프로 표현해 봅니다.

colormap = plt.cm.gist_heat   #그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12,12))   #그래프의 크기를 정합니다. 인치 단위다.

# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True) #vmax는 밝기 조절 인자. cmap은 정해진 matplotlib 색상의 설정 값을 불러옴.
plt.show()

grid = sns.FacetGrid(df, col='class') # 클래스별 plasma의 히스토그램.
grid.map(plt.hist, 'plasma',  bins=10)# bins은 10조각으로 나눈다는 의미 인듯.
plt.show() # 그래프를 겹쳐 본다면 당뇨병 환자의 경우(1) plasma 항목 수치가 150이상인 경우가 많음이 보임.
