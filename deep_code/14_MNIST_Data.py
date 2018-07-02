#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils

import numpy
import sys
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# MNIST데이터셋 불러오기 # MNIST는 60000을 학습용, 10000을 테스트용으로 미리 구분.
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data() # X_train란 변수에 이미지 저장. Y_class_train란 변수에 X_train에 해당하는 클래스. 뒤에도 비슷.

print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))  # shape는 행과 열을 튜플로 반환. 여기선 [0]이므로 행만 반환
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))
#print(X_train[0].flatten())
#print(X_train.shape[0])
# 그래프로 확인
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='Greys') # cmap='Greys'는 흑백 출력. 2차원에 숫자 뿐이라 농도로 나타내야해서 그렇다.
plt.show()# 28*28=784픽셀. 0~255까지 밝기가 있음.

# 코드로 확인 # 그러니까 각 그림이 어떤 숫자들로 이뤄져있는지 글자로 나타내 보이겠다는 거다.
for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 차원 변환 과정 # 1차원으로 바꾸는 과정. flatten 안쓰나?
X_train = X_train.reshape(X_train.shape[0], 784)  # (총 샘플수60000, 1차원 속성 수784) 즉 각 샘플마다 28픽셀 가로를 떼고 옆으로 붙이는 식으로 1줄씩. 6만번. #함수의 파라미터에 -1이 들어가면 특별한 의미를 갖는데, 다른 나머지 차원 크기를 맞추고 남은 크기를 해당 차원에 할당해 준다는 의미
X_train = X_train.astype('float64')
X_train = X_train / 255 # 행렬 안에 있는 값 하나하나 0~1사이 값이 되어 정규화.

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

#print(X_train[0])

# 클래스 값 확인
print("class : %d " % (Y_class_train[0]))

# 바이너리화 과정
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

print(Y_train[0])



