# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:17:18 2020

@author: SJL-PC
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

file_name="./boston_housing.npz"


def normalize(x,y):
    x = x - np.mean(x,axis=0)
    x = x/np.var(x,axis=0)

    y = y - np.mean(y, axis=0)
    y = y / np.var(y, axis=0)
    return x,y

def data_gen(file_name):
    datasets = np.load(file_name)
    x = datasets['x']
    y = datasets['y']
    x,y = normalize(x,y)
    list_idx = np.asarray(list(range(len(x))))
    np.random.seed(456)
    np.random.shuffle(list_idx)
    split = 0.2
    idx = int(len(x)*0.2)

    x_train = x[list_idx[:idx]]
    y_train = y[list_idx[:idx]]
    x_test = x[list_idx[idx:]]
    y_test = y[list_idx[idx:]]
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test) = data_gen(file_name)

y_train = np.expand_dims(y_train,axis=1)
y_test = np.expand_dims(y_test,axis=1)

x_t = tf.placeholder(dtype=tf.float32,shape=[None,13])
y_  = tf.placeholder(dtype=tf.float32,shape=[None,1])

#model
W = tf.Variable(tf.zeros([13,1]))#
b = tf.Variable(tf.zeros([1]))
y_pre = tf.matmul(x_t,W) + b

model_loss = tf.reduce_mean(tf.square(y_pre-y_))
# 调解优化器
optimazer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#optimazer = tf.train.AdamOptimizer(learning_rate=0.01)
train_opt = optimazer.minimize(loss=model_loss)

init = tf.global_variables_initializer()

loss_train_minibatch = []
loss_train_batch = []
loss_test = []
total_epochs = 20
with tf.Session() as sess:
    sess.run(init)

    for epochs in range(total_epochs):
        batch_size = 16
        total_batches = len(x_train)//batch_size
        if len(x_train) % batch_size >0:
            total_batches += 1
        for it in range(total_batches):
            x_train_batch = x_train[it * batch_size:min((it+1)*batch_size,len(x_train))]
            y_train_batch = y_train[it * batch_size:min((it + 1) * batch_size, len(y_train))]
            loss,_ = sess.run([model_loss,train_opt],feed_dict={x_t:x_train_batch,y_:y_train_batch})
            loss_train_minibatch.append(loss)


plt.plot(np.asarray(range(total_batches*total_epochs)),loss_train_minibatch,label='train_minibatch')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('sgd_test.png')  # 保存
plt.show()

