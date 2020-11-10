#coding:utf-8
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
sys.path.append("Game/")
import FlappyBirdtrain as game
import random
import numpy as np
from collections import deque

Game = 'bird'
Actions = 2 # 有两种动作0和1
Gamma = 0.99 # 对于过去行为动作的损失参数
Observe = 100000. # 前100000次不进行训练，只为获得样本动作
Explore = 2000000.
InitoalEpsilon = 0.0001 # 一开始的探索概率
FinalEpsilon = 0.0001 # 最终的探索概率
ReplayMemory = 50000 # 经验池中最多存放500000个数据样本
Batch = 32 # 一次选取32个数据样本用来更新网络
PerAction = 1

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (1, 80, 80))

def createNetwork():
    # 各层网络的参数
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, Actions])
    b_fc2 = bias_variable([Actions])

    # 输入
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 隐藏层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # 定义Q值函数
    a = tf.placeholder("float", [None, Actions])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 开始游戏
    Game_state = game.GameState()
    #一只鸟
    birds = []
    birds.append(game.Bird() )
    # birds.append(game.Bird() )
    # 将一开始得到的数据存入经验池D中
    D = deque()

    a_file = open("logs_" + Game + "/readout.txt", 'w')
    h_file = open("logs_" + Game + "/hidden.txt", 'w')

    # 在一开始的时候不做任何动作，但将第一帧的图片打包成80*80*4输入网络
    birds = Game_state.frame_step(birds)
    birds[0].screen = cv2.cvtColor(cv2.resize(birds[0].screen, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, birds[0].screen = cv2.threshold(birds[0].screen,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((birds[0].screen, birds[0].screen, birds[0].screen, birds[0].screen), axis=2)

    # 保存于加载网络
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 开始训练
    epsilon = InitoalEpsilon
    t = 0
    while True:
        for bird in birds:
            # 根据贪心策略选择所作的动作
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
            bird.action = np.zeros([Actions])
            action_index = 0
            if t % PerAction == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(Actions)
                    bird.action[random.randrange(Actions)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    bird.action[action_index] = 1
            else:
                bird.action[0] = 1

            # 缩小探索的概率
            if epsilon > FinalEpsilon and t > Observe:
                epsilon -= (InitoalEpsilon - FinalEpsilon) / Explore

        # 执行所选的动作，并且观察执行动作之后的状态和获得的奖励
        birds = Game_state.frame_step(birds)
        x_t1 = cv2.cvtColor(cv2.resize(birds[0].screen, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret,  x_t1 = cv2.threshold( x_t1, 1, 255, cv2.THRESH_BINARY)
        # x_t1=cv2.transpose(x_t1)
        # x_t1 = cv2.flip(x_t1, 0)
        # x_t1 = cv2.flip(x_t1, 0)
        # cv2.imshow("2", x_t1)
        x_t1 = np.reshape( x_t1, (80, 80, 1))
        s_t1 = np.append( x_t1, s_t[:, :, :3], axis=2)

        # 将数据样本保存在D中
        D.append((s_t, birds[0].action, birds[0].reward, s_t1, birds[0].endpoint))
        if len(D) > ReplayMemory:
            D.popleft()

        # 只有在观察之后才会进行新一次的训练
        if t > Observe:
            # 从D中选取Batch数量的样本更新miniBatch
            miniBatch = random.sample(D, Batch)

            # 获得(s,a,r,s)
            s_j_Batch = [d[0] for d in miniBatch]
            a_Batch = [d[1] for d in miniBatch]
            r_Batch = [d[2] for d in miniBatch]
            s_j1_Batch = [d[3] for d in miniBatch]

            y_Batch = []
            readout_j1_Batch = readout.eval(feed_dict = {s : s_j1_Batch})
            for i in range(0, len(miniBatch)):
                birds[0].endpoint = miniBatch[i][4]
                if birds[0].endpoint:
                    y_Batch.append(r_Batch[i])
                else:
                    y_Batch.append(r_Batch[i] + Gamma * np.max(readout_j1_Batch[i]))

            train_step.run(feed_dict = {
                y : y_Batch,
                a : a_Batch,
                s : s_j_Batch}
            )

        # 更新值
        s_t = s_t1
        t += 1

        # 没10000次训练保存一次网络
        if t % 100000 == 0:
            saver.save(sess, 'saved_networks/' + Game + '-dqn', global_step = t)
            break

        state = ""
        if t <= Observe:
            state = "Observe"
        elif t > Observe and t <= Observe + Explore:
            state = "Explore"
        else:
            state = "train"

        # print("TIMESTEP", t, " STATE", state, \
        #     "  EPSILON", epsilon, "  ACTION", action_index, "  REWARD", birds[0].reward, \
        #     "/ Q_MAX %e" % np.max(readout_t))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
