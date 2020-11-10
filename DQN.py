#coding:utf-8
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
sys.path.append("Game/")
import FlappyBird as game
import random
import numpy as np
from Net import *

Actions = 2 # 有两种动作0和1
epsilon = 0.0001 # 一开始的探索概率
PerAction = 1
# 开始游戏
Game_state = game.GameState()


def trainNetwork(bird,path):
    # 保存于加载网络
    saver = tf.train.Saver()
    bird.sess.run(tf.initialize_all_variables())
    saver.restore(bird.sess, path)
    tf.reset_default_graph()
    return bird


def setbird(path):
    bird = game.Bird()
    bird.sess = tf.InteractiveSession()
    bird.s, bird.readout, bird.h_fc1 = createNetwork()
    return trainNetwork(bird,path)

def getbirds():
    checkpoint = tf.train.get_checkpoint_state("saved_networks/")
    birds = []
    for i in range(len(checkpoint.all_model_checkpoint_paths)):
        path = checkpoint.all_model_checkpoint_paths[i]
        bird = setbird(path)
        bird.path=path
        bird.flag=i+1
        bird.color = (random.randint(0, 255) , random.randint(0, 255) , random.randint(0, 255))
        birds.append(bird)

    # 在一开始的时候不做任何动作，但将第一帧的图片打包成80*80*4输入网络
    birds = Game_state.frame_step(birds)   
    birds[0].screen = cv2.cvtColor(cv2.resize(birds[0].screen, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, birds[0].screen = cv2.threshold(birds[0].screen,1,255,cv2.THRESH_BINARY)
    for bird in birds:
        bird.s_t = np.stack((birds[0].screen, birds[0].screen, birds[0].screen, birds[0].screen), axis=2)

    return birds

def main():
    birds = getbirds()
    birds[0].end=False
    t = 0
    print("----------竞赛开始----------")
    for bird in birds:
        print(str(bird.flag)+"号小鸟  对应模型："+bird.path)
    print("----------共"+str(len(birds))+"只小鸟----------")
    while True:
        if birds[0].end:
            break
        for bird in birds:
            # 根据贪心策略选择所作的动作
            readout_t = bird.readout.eval(session=bird.sess,feed_dict={bird.s : [bird.s_t]})[0]
            bird.action = np.zeros([Actions])
            action_index = 0
            if t % PerAction == 0:
                if random.random() <= epsilon:
                    action_index = random.randrange(Actions)
                    bird.action[random.randrange(Actions)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    bird.action[action_index] = 1
            else:
                bird.action[0] = 1


        # 执行所选的动作，并且观察执行动作之后的状态和获得的奖励
        birds = Game_state.frame_step(birds)
        
        for bird in birds:
            x_t1 = cv2.cvtColor(cv2.resize(bird.screen, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret,  x_t1 = cv2.threshold( x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape( x_t1, (80, 80, 1))
            s_t1 = np.append( x_t1, bird.s_t[:, :, :3], axis=2)

            # 更新值
            bird.s_t = s_t1
        t += 1

if __name__ == "__main__":
    main()
