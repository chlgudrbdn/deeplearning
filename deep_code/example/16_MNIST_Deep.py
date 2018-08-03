#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255  #좀 독특해보이는 reshape인데 4차원으로 각 샘플마다 픽셀하나하나 개별적인 리스트형으로 만들어버림. 합성곱을 위한 과정으로 보인다.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # 10000x28x28x1 각 샘플마다
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))# 마스크(=필터, 커널, 윈도) 합성곱 해버리면 특징이 더 정교해진다. 이렇게 만들어진 새로운 층을 컨볼루션(합성곱)이라 부른다.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) # 풀링 층 추가. 앞서 구현한 컨볼루션 층으로 특징을 추출했는데 결과가 여전히 크고 복잡하면 이를 축소하는 것을 풀링(=서브 샘플링)이라 한다. 그중 max 풀링을 자주 사용. 정해진 구역 안에서 가장 큰 값 남기고 나머지는 넘기는 방식. 풀링 사이즈가 2니까 전체 크기는 절반이 될것.
model.add(Dropout(0.25)) # 노드, 층이 많다고 성능이 좋아지는 건 아님. 이걸 돕는 기법중 하나인 드롭아웃. 은닉층 노드 중 일부를 임의로 꺼주는 것. 지나치게 치우쳐 학습하는 것을 방지.
model.add(Flatten()) # 위의 과정은 2차원 배열을 다룬 것이므로 이를 1차원 함수로 바꿔주는 함수.
model.add(Dense(128,  activation='relu')) #책에선 그림으로 512개의 노드라 되어 있긴한데 신경쓰지 말자.
model.add(Dropout(0.5)) #  한번 더함.
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

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
