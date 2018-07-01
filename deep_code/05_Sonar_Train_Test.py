from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('../dataset/sonar.csv', header=None)

print(df.info())
print(df.head())

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습 셋과 테스트 셋의 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
# http://data-newbie.tistory.com/31 괜찮게 정리해뒀다.
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])
# 은닉 층이 늘어날 수록 tarin 전체에 대한 예측률은 올라가지만 test에 대한 예측률은 도리어 떨어질 수 있다.
# https://thebook.io/006958/part04/ch13/03-01/의 그림을 보면 테스트 에러가 가장 적을 때 과적합이 발생할 가능성이 적어짐. 반대라면 과적합 발생.
model.fit(X_train, Y_train, epochs=130, batch_size=5)

# 테스트셋에 모델 적용
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# 실전에서는 더 정확한 테스트를 위해 테스트셋을 두 개로 나누어, 하나는 앞서 설명한 방식대로 테스트셋으로 사용하고, 나머지 하나는 최종으로 만들어 낸 모델을 다시 한번 테스트하는 용도로 사용하기도 합니다. 추가로 만들어낸 테스트셋을 검증셋(Validation sets)이라고도 부릅니다.
