#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-03-08 16:57
@edit time: 2020-03-08 20:14
@FilePath: /asv_dynamic.py
"""
import numpy as np


class Dim2Object(object):
    def __init__(self, desc):
        self.description = desc
        self.pos = np.array([0.0, 0.0])

    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, nx):
        self.pos[0] = nx

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, ny):
        self.pos[1] = ny

    @property
    def data(self):
        return self.pos.copy()

    def __str__(self):
        return f'x: {self.x}, y: {self.y}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class ASV(object):
    """简易潜艇模型的对象
    一艘潜艇的状态，包括其：
    坐标、速度和加速度， 且都是二维的
    """

    def __init__(self, time_interval):
        self.time_interval = time_interval

        self.__position = Dim2Object('position')  # 单位是px
        self.__velocity = Dim2Object('velocity')  # 单位是px/s
        self.__acceleration = Dim2Object('acceleration')  # 单位是px/s^2

        self.__max_velocity = 120
        self.__max_acceleration = 600

    @property
    def position(self):
        return self.__position

    def reset_state(self):
        """重设潜艇位置、速度、加速度为(0, 0)"""
        self.__position.x, self.__position.y = 0, 0
        self.velocity = (0, 0)
        self.acceleration = (0, 0)
        return self.__position

    def rand_state(self, xrange, yrange):
        """在指定范围内随机设置潜艇位置，重置速度、加速度"""
        self.__position.x = np.randint(*xrange)
        self.__position.y = np.randint(*yrange)
        self.velocity = (0, 0)
        self.acceleration = (0, 0)
        return self.__position

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, v):
        """如果是直接设置速度，那应该要有上限
        按照5s移动完整个屏幕计算，600px为正常尺寸，则
        max velocity = 120
        """
        v_x, v_y = np.clip(v, -self.max_velocity, self.max_velocity)
        self.__velocity.x, self.__velocity.y = v_x, v_y

    @property
    def acceleration(self):
        return self.__acceleration

    @acceleration.setter
    def acceleration(self, a):
        """加速度显然也应该有上限
        加速到最大速度的时间设置为2s，则
        max acceleration = 60
        """
        a_x, a_y = np.clip(a, -self.max_acceleration, self.max_acceleration)
        self.__acceleration.x, self.__acceleration.y = a_x, a_y

    @property
    def max_velocity(self):
        """没有setter，这应该是只读的"""
        return self.__max_velocity

    @property
    def max_acceleration(self):
        """没有setter，这应该是只读的"""
        return self.__max_acceleration

    def move(self):
        """计算asv下一步的位置
        分别对于x和y两个方向，计算位移
        """
        pt = self.time_interval
        cur_pos = self.position
        v_cur, a_cur = self.velocity, self.acceleration
        v_max = self.max_velocity

        # 计算x方向
        a, v = a_cur.x, v_cur.x
        if a > 0:
            # 若x方向加速度是正的，则考虑加速到最大速度需要多少时间
            acc_dur = (v_max - v) / a  # 得到最长加速时间
            if acc_dur > pt:
                # 如果加速到最大需要的时间超过interval，说明整个interval内都可以加速
                # s = v*t + 1/2 * a*t^2
                delta_x = v * pt + 0.5 * a * pt * pt
                # 同时，时间结束之后速度变为 v + a * t
                vx_end = v + a * pt
            else:
                # 否则说明加速一段时间之后就会达到最大速度
                # s = s1(匀加速) + s2(匀速)
                # s1 = 1/2 * (v_0 + v_1) * t1, 式中v_0 = v_cur
                # s2 = v_1 * (t - t1), 式中v_1 = v_max
                delta_x = (v + v_max) * acc_dur / 2 + v_max * (pt - acc_dur)
                # 同时，时间结束之后速度变为 v_max
                vx_end = v_max
        elif a < 0:
            # 若x方向加速度是负的，则考虑加速到负的最大速度时长
            # 注意 -vmax<v ， a<0 所以acc_dur依然是正的
            acc_dur = (-v_max - v) / a  # 得到最长加速时间
            if acc_dur > pt:
                # 如果加速到最大需要的时间超过interval，说明整个interval内都可以加速
                # s = v*t + 1/2 * a*t^2
                delta_x = v * pt + 0.5 * a * pt * pt
                # 同时，时间结束之后速度变为 v + a * t
                vx_end = v + a * pt
            else:
                # 否则说明加速一段时间之后就会达到最大速度
                # s = s1(匀加速) + s2(匀速)
                # s1 = 1/2 * (v_0 + v_1) * t1, 式中v_0 = v_cur
                # s2 = v_1 * (t - t1), 式中v_1 = - v_max
                delta_x = (v - v_max) * acc_dur / 2 - v_max * (pt - acc_dur)
                # 同时，时间结束之后速度变为 -v_max
                vx_end = -v_max
        else:
            # 如果x方向没有加速度，则是匀速运动
            # s = v*t
            delta_x = v * pt
            vx_end = v

        # 计算y方向， 同理
        a, v = a_cur.y, v_cur.y
        if a > 0:
            acc_dur = (v_max - v) / a  # 得到最长加速时间
            if acc_dur > pt:
                delta_y = v * pt + 0.5 * a * pt * pt
                vy_end = v + a * pt
            else:
                delta_y = (v + v_max) * acc_dur / 2 + v_max * (pt - acc_dur)
                vy_end = v_max
        elif a < 0:
            acc_dur = (-v_max - v) / a  # 得到最长加速时间
            if acc_dur > pt:
                delta_y = v * pt + 0.5 * a * pt * pt
                vy_end = v + a * pt
            else:
                delta_y = (v - v_max) * acc_dur / 2 - v_max * (pt - acc_dur)
                vy_end = -v_max
        else:
            delta_y = v * pt
            vy_end = v
        self.__position.x += delta_x
        self.__position.y += delta_y
        self.velocity = (vx_end, vy_end)
        return self.position
