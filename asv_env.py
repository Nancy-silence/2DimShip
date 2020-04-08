#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-01-16 10:00
@edit time: 2020-03-09 16:39
@FilePath: /asv/asv_env.py
"""
from asv_dynamic import ASV
from move_point import MovePoint
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class ASVEnv(gym.Env):
    """
    ASV 的环境
    使用一个asv去追逐一个点
    """

    def __init__(self, action_type='velocity', target_trajectory='linear', interval=0.1):
        self.action_type = action_type
        self.target_trajectory = target_trajectory
        self.interval = interval
        self.asv = ASV(self.interval)
        self.aim = MovePoint(self.target_trajectory)
        self.playground_shape = (-10, 120, -70, 70)

        plt.ion()
        self.aim_his = [self.aim.position]
        self.asv_his = [self.asv.position.data]

        self.observation_space = spaces.Box(low=0, high=600, shape=(4,))
        self.action_space = spaces.Box(low=0, high=600, shape=(2,))

    def reset(self):
        """重设环境状态
        将目标点重置到(0, 0)之后，获取下一个目标点
        将船只重置为(0, 0)
        reset返回的state是当前点与目标点的差值
        """
        self.aim.reset()
        self.aim.next_point(self.interval)
        self.asv.reset_state()
        aim_pos = self.aim.position
        asv_pos = self.asv.position.data
        self.aim_his = [list(aim_pos)]
        self.asv_his = [list(asv_pos)]
        plt.ioff()
        plt.clf()
        plt.ion()
        return self.get_state()

    def get_state(self):
        """获取当前环境状态，即目标点坐标与船只坐标的差值"""
        asv_pos = self.asv.position.data
        aim_pos = self.aim.position
        state = np.append(aim_pos - asv_pos, self.asv.velocity.data)
        return state

    def get_reward(self):
        asv_pos = self.asv.position.data
        aim_pos = self.aim.position
        d = np.sum(np.power((asv_pos - aim_pos), 2))
        r = np.exp(-d/100)-1
        return r

    def get_reward_punish(self):
        return -5

    def get_done(self):
        if (self.asv.position.x < self.playground_shape[0] or self.asv.position.x > self.playground_shape[1] or
                self.asv.position.y < self.playground_shape[2] or self.asv.position.y > self.playground_shape[3]):
            return True
        return False

    def step(self, action):
        if self.action_type == 'velocity':
            self.asv.velocity = action
        elif self.action_type == 'acceleration':
            self.asv.acceleration = action
        else:
            raise TypeError("不在列表中的动作类型")

        # 注意因为reset中已经让aim移动，因此aim永远是asv要追逐的点
        # 在获得action之后，让asv根据action移动
        # 奖励应该是对于当前aim，以及移动以后的asv计算
        # 让asv移动，则当前asv坐标更新为移动后的坐标
        cur_asv = self.asv.move()
        done = self.get_done()
        # 注意奖励永远是根据当前aim坐标和当前asv坐标计算，当前aim尚未移动
        if done:
            reward = self.get_reward_punish()
        else:
            reward = self.get_reward()
        # 计算完奖励之后，可以移动aim坐标
        cur_aim = self.aim.next_point(self.interval)
        # 此时aim已经是下一个要追逐的点，可以计算state
        state = self.get_state()

        # 记录坐标点，便于绘图
        self.aim_his.append(list(cur_aim))
        self.asv_his.append(list(cur_asv.data))

        return state, reward, done, ''

    def render(self):
        plt.clf()
        # plt.xlim(-1, )
        # plt.ylim(-1, 200)
        # 绘制aim
        plt.plot(*zip(*self.aim_his), 'y')

        # 绘制asv
        plt.plot(*zip(*self.asv_his), 'b')

        plt.pause(0.1)
