# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 실행할 때마다 같은 결과를 출력하기 위한 seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# x,y의 데이터 값
x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)

# 입력 값을 플래이스 홀더에 저장. placeholder는 입력값을 저장하는 일종의 그릇으로 tf.placeholder("데이터형", "행렬의 차원", "이름")형태로 사용.
X = tf.placeholder(tf.float64, shape=[None, 2])#[xxxx xxxx] 같은 식으로 저장된단 소리. 데이터 주입 통로로 연산노드를 가리키는 텐서
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기 a와 bias b의 값을 임의로 정함.
a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))  # [2,1] 의미: 들어오는 값은 속성 X의 종류 2개, 나가는 값은 1개. 들어오고 나간다는건 행렬 곱한단 얘기다.
# 예시로 a = tf.random_uniform([2,1],seed=42)이면 a= [[0.95227146] [0.67740774]]느낌으로 1의 길이 리스트 2개가 1개로 묶여 나온다. 즉 2행 1열.
# [2,3]이면 3개의 숫자를 가진 두개의 리스트가 하나로 묶여 나온다. 즉 2행 3열 https://zetawiki.com/wiki/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0_random_uniform() 참고
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# y 시그모이드 함수의 방정식을 세움
y = tf.sigmoid(tf.matmul(X, a) + b)

# 오차를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# 학습률 값
learning_rate = 0.1

# 오차를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64) # cast는 데이터 타잎 바꾼단 의미. 여기선 0.5이상일 경우
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))

    # 어떻게 활용하는가
    new_x = np.array([7, 6.]).reshape(1, 2)  # [7, 6]은 각각 공부 시간과 과외 수업수.
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부 시간: %d, 개인 과외 수: %d" % (new_x[:, 0], new_x[:, 1]))
    print("합격 가능성: %6.2f %%" % (new_y * 100))
