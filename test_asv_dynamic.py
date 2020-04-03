#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-03-08 16:56
@edit time: 2020-03-09 16:25
@FilePath: /asv/test_asv_dynamic.py
"""


import pytest
import numpy as np
from asv_dynamic import ASV


class TestASV():
    @classmethod
    def setup_class(cls):
        """创建时间间隔为1s的ASV实例"""
        cls.interval = 2
        cls.asv = ASV(cls.interval)

    def setup(self):
        """每次测试前重置asv
        位置重置为(0, 0)， 速度重置为(0, 0)
        interval重置为最开始的interval"""
        self.asv.reset_state()
        self.asv.interval = self.interval

    def test_pos_getter(self):
        """测试move()返回的pos和获取的pos是否一致"""
        asv = self.asv
        # 设置速度
        rand_v = np.random.randint(0, 100, 2)
        asv.velocity = rand_v
        # 设置加速度
        rand_a = np.random.randint(0, 10, 2)
        asv.acceleration = rand_a
        # 检测位置
        rt_pos = asv.move()
        gt_pos = asv.position
        assert rt_pos == gt_pos

    def test_uniform_velocity(self):
        """测试匀速直线运动
        从点(0, 0)开始，以vx=10, vy=-10做匀速直线运动
        一个时间间隔过后，以vx=-5, vy=15做匀速直线运动
        """
        asv, interval = self.asv, self.interval
        # 第一次设置速度
        asv.velocity = (10, -10)
        first_pos = asv.move()
        # 当前位置应该为(10, -10)
        assert first_pos.x == 10 * interval
        assert first_pos.y == -10 * interval
        # 第二次设置速度
        asv.velocity = (-5, 15)
        second_pos = asv.move()
        # 当前位置应该为(5, 5)
        assert second_pos.x == 5 * interval
        assert second_pos.y == 5 * interval

    def test_velocity_exceed(self):
        """测试超速
        将速度设置为速度上限的2倍，看是否会被执行
        """
        asv, interval = self.asv, self.interval
        # 获取速度上限
        v_max = asv.max_velocity
        # 设置速度
        asv.velocity = (v_max * 2, 0)
        # 检查速度
        assert asv.velocity.x == v_max
        # 检查运行状态
        cur_pos = asv.move()
        assert cur_pos.x == v_max * interval

    def test_uniform_acceleration_symbol(self):
        """测试匀加速直线运动，确保在正负数下计算都正确"""
        asv, interval = self.asv, self.interval
        # 设置加速度
        asv.acceleration = (30, -20)
        # 测试位置
        cur_pos = asv.move()
        assert cur_pos.x == 15 * interval**2
        assert cur_pos.y == -10 * interval**2
        # 测试速度
        v = asv.velocity
        assert v.x == 30 * interval
        assert v.y == -20 * interval

        # 设置加速度
        asv.acceleration = (-30, 20)
        # 测试位置
        cur_pos = asv.move()
        assert cur_pos.x == 30 * interval**2
        assert cur_pos.y == -20 * interval**2
        # 测试速度
        v = asv.velocity
        assert v.x == 0
        assert v.y == 0

    def test_uniform_acceleration_exceed(self):
        """测试加速度溢出的情况"""
        asv, interval = self.asv, self.interval
        # 将速度设置为接近超速
        a_max = asv.max_acceleration
        # 设置加速度
        asv.acceleration = (2 * a_max, -2 * a_max)
        # 测试加速度
        a = asv.acceleration
        assert a.x == a_max
        assert a.y == -a_max
        # 测试速度
        asv.move()
        v = asv.velocity
        assert v.x == a_max * interval
        assert v.y == -a_max * interval
        # 测试位移
        p = asv.position
        assert p.x == 0.5 * a_max * interval**2
        assert p.y == -0.5 * a_max * interval**2

    def test_acceleration_to_exceed_speed(self):
        """测试加速到超速
        从离超速1开始加速，加速度1，则有1s在加速,剩下的时间匀速
        先测x+y-，重置后测x-y+，覆盖所有情况
        """
        asv = self.asv
        asv.time_interval = 2    # 使用interval = 2进行测试
        v_max = asv.max_velocity
        assert asv.max_acceleration >= 1    # 如果a_max < 1，则无法测试
        # 设置速度
        asv.velocity = (v_max - 1, -(v_max - 1))
        # 设置加速度
        asv.acceleration = (1, -1)
        # 测试速度、位置
        asv.move()
        v = asv.velocity
        assert v.x == v_max
        assert v.y == -v_max
        p = asv.position
        assert p.x == v_max * 2 - 0.5
        assert p.y == -(v_max * 2 - 0.5)

        # 重置，重新测试
        asv.reset_state()
        # 设置速度
        asv.velocity = (-(v_max - 1), v_max - 1)
        # 设置加速度
        asv.acceleration = (-1, 1)
        # 测试速度、位置
        asv.move()
        v = asv.velocity
        assert v.x == -v_max
        assert v.y == v_max
        p = asv.position
        assert p.x == -(v_max * 2 - 0.5)
        assert p.y == v_max * 2 - 0.5