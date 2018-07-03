# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 로이터 뉴스 데이터셋 불러오기. 케라스가 제공하는 걸로 RNN 학습에 적절한 텍스트 대용량 자료.
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 불러온 데이터를 학습셋, 테스트셋으로 나누기
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2) # num_words=1000는 빈도가 1~1000에 해당하는 단어만 선택해서 부른다는 의미.

# 데이터 확인하기
category = numpy.max(Y_train) + 1 # +1은 0부터 세서
print(category, '카테고리')
print(len(X_train), '학습용 뉴스 기사')
print(len(X_test), '테스트용 뉴스 기사')
print(X_train[0]) # 단어를 그대로 쓸 수 없어 숫자로 변환해야함. 단어의 빈도를 세어서 번호를 붙인 방식. 3이면 세번째로 빈도가 높은 단어란 소리. 거의 쓰지 않는건 별 의미 없어서(?? 그럴 수록 중요한 케이스도 있지만서도.)

# 데이터 전처리 # 각 기사의 단어수가 제각각이면 행렬곱이 어려워서 사용.
x_train = sequence.pad_sequences(X_train, maxlen=100) # 만약 입력된 기사의 단어 수가 100보다 크면 100개째 단어만 선택. 나머지는 버리는 방식. 100보다 작으면 모자라는 건 모두 0으로 채운다.
x_test = sequence.pad_sequences(X_test, maxlen=100)
y_train = np_utils.to_categorical(Y_train) # 클래스는 원핫 인코딩 처리.
y_test = np_utils.to_categorical(Y_test)

# 모델의 설정
model = Sequential()
model.add(Embedding(1000, 100)) # Embedding층은 데이터 전처리 과정 통해 입력된 값을 받아 다음 층이 알아들을 수 있는 형태로 변환하는 역할. (불러온 단어의 총 개수, 기사당 단어 수). 1000가지 단어를 각 샘플마다 100개씩 feature로 갖고 있다. ??? 왜 이런 형태인지 이해가 안간다. 부록 보고 보충바람???
model.add(LSTM(100, activation='tanh'))# RNN은 일반 신경망 보다 기울기 소실 문제가 더 많이 발생하고 이를 해결하기 어렵다는 단점을 보완하기 위해 LSTM사용. 즉 반복 되기 직전에 다음층으로 기억된 값을 넘길지 안넘길지 관리하는 단계 하나 더 추가. 기억값에 대한 가중치를 제어하는 방식. Dense나 CNN 대신 RNN의 방식 중 하나.
# tanh로 한 이유는 딱히 이유는 없는 것 같다.
model.add(Dense(46, activation='softmax')) # RNN은 여러 상황에서 쓰일수 있는데 다수 입력 단일 출력, 단일 입력 다수 출력도 가능(사진의 여러 요소를 추출해 캡션 만들 때 사용). 이건 후자. 다수입력 다수 출력도 가능.

# 모델의 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))


# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
