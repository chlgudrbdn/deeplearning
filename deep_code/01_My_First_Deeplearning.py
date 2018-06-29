# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17] # 모든 행 : , 0~17사이 열
Y = Data_set[:,17] # 한 470줄 된다.

# 딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential() # 구조를 쌓아갈 수 있도록 하는 함수. add함수를 쓰면 층이 추가됨.
model.add(Dense(30, input_dim=17, activation='relu'))# Dense 함수는 조밀하게 모여있는 집합, 구조을 의미. input_dim=17 입력층, 30은 은닉층 노드 개수. 두 층의 역할을 겸함.
model.add(Dense(1, activation='sigmoid'))# 마지막은 출력층으로서 사용. 1인건 결국 0 아님 1 하나만 출력해야해서.
# activation:다음 층으로 넘기는 활성함수 지정 방법. sigmoid인건 양자 택일을 위함.
# 딥러닝을 실행합니다. 위에서 정해진 모델을 컴퓨터가 알아들을 수 있게끔 컴파일 하기 위함.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # metrics:모델 수행결과를 나타나게끔 설정하는 부분. 테스트 기능을 담고 있음.
#loss:한변 신경망 실행될 때 마다 오차값 추정 함수, Optimizer: 오차 줄여나갈 방법
# mean_squared_error 평균 제곱오차 계열은 수렴에 속도가 많이걸린다는 단점이 있음.
# 교차 엔트로피 계열 함수는 출력에 로그를 취해서 오차가 커지면 수렴 속도가 빨라지고, 오차가 작아지면 속도가 감소하게끔 만드는 방식.
# 주로 분류에 많이 쓰이는데, 이항의 결과는 이항 교차 엔트로피(binary_crossentropy)를 씀. 이 예시에서 제곱오차 대신 쓰면 성능이 약간 상승함.
model.fit(X, Y, epochs=30, batch_size=10) # 실행. 1 epochs는 모든 데이터셋을 학습한다는것. 30이면 각 샘플(행)이 처음부터 끝까지 30번 재사용 될 떄까지 실행을 반복하란 것.
# 1epoch=전체데이터 개수/batch size= 470/10=47iteration
# batch size는 샘플을 몇개씩 끊어서 집어넣는다는 의미. 너무 크면 속도가 느려지고 작으면 실행값의 편차가 생겨 전체 결과값이 불안정해질 수 있다. 메모리가 감당할 수준이어야함.

# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))


