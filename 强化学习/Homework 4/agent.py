#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

# agent实现，包含训练和预测

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
import math
import random
from dqn.model.model import Model
from dqn.feature.definition import ActData
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from dqn.config import Config


class LearningAlgo():

    def __init__(self,
                 lr,
                 device,
                 obs_shape,
                 act_shape,
                 softmax,
                 _gamma,
                 target_update_freq: int = 50,
                 algo_name: str = 'DQN'):
        """ algo_name is one of `DQN`, `TARGET_DQN`. You can also implement
            other algorithms, e.g. `DOUBLE_DQN`, if you will. """
        self.lr = lr
        self.device = device
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.softmax = softmax
        self._gamma = _gamma
        self.dueling = True

        self.model = Model(state_shape=self.obs_shape,
                           action_shape=self.act_shape,
                           softmax=self.softmax,
                           dueling=self.dueling)  # 是 evaluation_model
        self.model.to(self.device)  # 将模型移动到指定设备上
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.lr)  # 优化器
        self.algo_name = algo_name
        if self.algo_name == 'TARGET_DQN' or self.algo_name == 'DOUBLE_DQN':
            self.target_update_freq = target_update_freq  # 更新频率
            """ Coding: Define your target model here. """
            # 定义 Target DQN 和 Double DQN 的目标模型
            self.target_model = Model(state_shape=self.obs_shape,
                                      action_shape=self.act_shape,
                                      softmax=self.softmax,
                                      dueling=self.dueling)
            self.target_model.to(self.device)  # 无梯度，每隔一段时间更新成model
            # self.target_model.load_state_dict(self.model.state_dict())
            self.train_step = 0
            """ End Coding """

    def target_q(self, reward, _batch_feature, _batch_legal_actions, not_done):
        """ Coding: calculate the target Q value for DQN and TARGET_DQN here.
            Hint: you have to ensure that the algorithm chooses a legal action
            (_batch_legal_actions). You can refer to the platform's
            implementation. """
        # 计算 DQN 和 TARGET_DQN 的目标 Q 值
        # 必须确保算法选择一个合法的动作 (_batch_legal_actions)，可以参考平台实现
        # print(type(_batch_legal_actions))  # torch.tensor
        # print(_batch_legal_actions)  # bool
        # batch_legal_actions = torch.tensor(_batch_legal_actions,dtype=torch.int32).to(self.device)
        not_dones = torch.tensor(not_done,
                                 dtype=torch.float).view(-1, 1).to(self.device)
        reward = torch.tensor(reward,
                              dtype=torch.float).view(-1, 1).to(self.device)
        print('reward: ', reward)
        # 获得q_max
        with torch.no_grad():
            # if double dqn, choose a' with model, else with target_model
            if self.algo_name == 'DOUBLE_DQN':
                # 用model选action，用target_model计算Q
                q, _ = self.model(_batch_feature, state=None)  # get all q(s')
                q = q.masked_fill(~_batch_legal_actions,
                                  float(torch.min(q) - 1))  # 填充不合法的动作为minq-1
                max_indices = q.max(dim=1).indices.detach()
                qt, _ = self.target_model(_batch_feature,
                                          state=None)  # get all q(s')
                qt = qt.masked_fill(~_batch_legal_actions,
                                    float(torch.min(qt) -
                                          1))  # 填充不合法的动作为minq-1
                # 根据indices返回最大Q（应该是二维的，dim=1是action）
                # print('qt: ', qt)
                q_max = torch.gather(
                    qt,
                    dim=1,
                    index=max_indices.unsqueeze(
                        1)  # 变成 (-1, 1) 以匹配 target_tensor 的 dim=1
                )
            elif self.algo_name == 'TARGET_DQN':
                # 返回target_model的最大Q
                q, h = self.target_model(_batch_feature,
                                         state=None)  # get all q(s')
                q = q.masked_fill(~_batch_legal_actions,
                                  float(torch.min(q) - 1))  # 填充不合法的动作为minq-1
                q_max = q.max(dim=1).values.detach()
            elif self.algo_name == 'DQN':
                # 没有目标网络，直接返回model的最大Q
                q, h = self.model(_batch_feature, state=None)  # get all q(s')
                q = q.masked_fill(~_batch_legal_actions,
                                  float(torch.min(q) - 1))  # 填充不合法的动作为minq-1
                q_max = q.max(dim=1).values.detach()
        #print('q_max:', q_max)
        target_q = reward + self._gamma * q_max * not_dones  # (-1,1)
        """ End Coding """
        #print('target_q: ', target_q)

        return target_q

    def learn(self,
              batch_feature_vec,
              batch_feature_map,
              _batch_feature_vec,
              _batch_feature_map,
              batch_action,
              batch_not_done,
              batch_reward,
              _batch_legal_actions,
              get_monitoring_info: bool = True):
        """ You can consider batch_feature as current state, and 
            _batch_feature as next_state. Calculate target Q values as
            estimator of ground truth Q values. """

        # batch_action 动作
        # batch_not_done 奖励？
        # batch_reward 奖励
        # _batch_legal_actions 是否完成（done）?
        # 为什么顺序和definiation里面不一样？

        # 把 batch_feature 看作当前状态, _batch_feature 看作下一个状态.
        # 计算目标 Q 值作为真实 Q 值的估计值
        batch_feature = [
            batch_feature_vec,
            batch_feature_map,
        ]  # 全部的obs，当成当前状态
        _batch_feature = [
            _batch_feature_vec,
            _batch_feature_map,
        ]  # 全部的obs_legal，当成下一个状态
        target_q = self.target_q(batch_reward, _batch_feature,
                                 _batch_legal_actions, batch_not_done)
        """ Coding: execute training loop. Use the previously calculated
            target_q to calculate loss and update target network. """
        # 执行训练循环. 使用之前计算的 target_q 来计算损失并更新目标网络
        self.optim.zero_grad()
        q_value, h = self.model(batch_feature, state=None)
        # 计算均方误差（从这里，Q大概都是二维的）
        loss = torch.square(
            target_q - q_value.gather(1, batch_action).view(-1, 1)).mean()
        loss.backward()
        # 裁剪模型梯度防止梯度爆炸
        # model_grad_norm =
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.optim.step()

        # value_loss = loss.detach().item()
        # q_value = target_q.mean().detach().item()
        # reward = batch_reward.mean().detach().item()

        if self.algo_name == 'DOUBLE_DQN' or self.algo_name == 'TARGET_DQN':
            if self.train_step % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            self.train_step += 1
        """ End Coding. """

        if get_monitoring_info:
            value_loss = loss.detach().item()
            q_value = target_q.mean().detach().item()
            reward = batch_reward.mean().detach().item()
            print(
                f'Learn function called. Value loss {value_loss}, Q value {q_value}, reward {reward}.'
            )

            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": 0,
                "diy_2": 0,
                "diy_3": 0,
                "diy_4": 0,
                "diy_5": 0,
            }
            return monitor_data
        return


@attached
class Agent(BaseAgent):

    def __init__(self,
                 agent_type="player",
                 device=None,
                 logger=None,
                 monitor=None,
                 double_dqn: bool = True,
                 replay_buffer: bool = True):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR

        self.device = device
        self.algo = LearningAlgo(self.lr, self.device, self.obs_shape,
                                 self.act_shape, False, self._gamma)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                data,
                device=self.device,
                dtype=torch.float32,
            )

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [
            obs_data.feature[:self.obs_split[0]] for obs_data in list_obs_data
        ]
        feature_map = [
            obs_data.feature[self.obs_split[0]:] for obs_data in list_obs_data
        ]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (torch.cat(
            (
                legal_act[:, 0].unsqueeze(1).expand(batch,
                                                    self.direction_space),
                legal_act[:, 1].unsqueeze(1).expand(batch,
                                                    self.talent_direction),
            ),
            1,
        ).bool().to(self.device))
        model = self.algo.model
        model.eval()
        # Exploration factor,
        # we want epsilon to decrease as the number of prediction steps increases, until it reaches 0.1
        # 探索因子, 我们希望epsilon随着预测步数越来越小，直到0.1为止
        self.epsilon = max(0.1, self.epsilon - self.predict_count / self.egp)

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action,
                                             dtype=torch.float32).to(
                                                 self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(
                        batch, *self.obs_split[1]),
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act,
                                            float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[
            instance[0] % self.direction_space,
            instance[0] // self.direction_space
        ] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    def get_batch_from_sample(self,
                              t_data,
                              obtain_next_legal_actions: bool = True):
        batch_len = len(t_data)
        batch_feature_vec = [frame.obs[:self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0]:] for frame in t_data]
        _batch_feature_vec = [
            frame._obs[:self.obs_split[0]] for frame in t_data
        ]
        _batch_feature_map = [
            frame._obs[self.obs_split[0]:] for frame in t_data
        ]
        batch_action = [int(frame.act) for frame in t_data]
        batch_not_done = [0 if frame.done == 1 else 1 for frame in t_data]
        batch_reward = [frame.rew for frame in t_data]

        batch_feature_vec = self.__convert_to_tensor(batch_feature_vec)
        batch_feature_map = self.__convert_to_tensor(batch_feature_map).view(
            batch_len, *self.obs_split[1])
        _batch_feature_vec = self.__convert_to_tensor(_batch_feature_vec)
        _batch_feature_map = self.__convert_to_tensor(_batch_feature_map).view(
            batch_len, *self.obs_split[1])
        batch_action = torch.LongTensor(np.array(batch_action)).view(-1, 1).to(
            self.device)
        batch_not_done = torch.tensor(np.array(batch_not_done),
                                      device=self.device)
        batch_reward = torch.tensor(np.array(batch_reward), device=self.device)

        return_items = [
            batch_feature_vec,
            batch_feature_map,
            _batch_feature_vec,
            _batch_feature_map,
            batch_action,
            batch_not_done,
            batch_reward,
        ]

        if obtain_next_legal_actions:
            _batch_obs_legal = torch.tensor(
                np.array([frame._obs_legal for frame in t_data]))
            _batch_obs_legal = (torch.cat(
                (
                    _batch_obs_legal[:, 0].unsqueeze(1).expand(
                        batch_len, self.direction_space),
                    _batch_obs_legal[:, 1].unsqueeze(1).expand(
                        batch_len, self.talent_direction),
                ),
                1,
            ).bool().to(self.device))
            return_items.append(_batch_obs_legal)

        return return_items

    @learn_wrapper
    def learn(self, list_sample_data):

        # get training data from collected episode
        batch_items = self.get_batch_from_sample(
            list_sample_data, obtain_next_legal_actions=True)

        batch_feature_vec = batch_items[0]
        batch_feature_map = batch_items[1]
        _batch_feature_vec = batch_items[2]
        _batch_feature_map = batch_items[3]
        batch_action = batch_items[4]
        batch_not_done = batch_items[5]
        batch_reward = batch_items[6]
        _batch_legal_actions = batch_items[7]

        monitor_data = self.algo.learn(
            batch_feature_vec,
            batch_feature_map,
            _batch_feature_vec,
            _batch_feature_map,
            batch_action,
            batch_not_done,
            batch_reward,
            _batch_legal_actions,
            get_monitoring_info=True,
        )
        self.train_step += 1

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {
            k: v.clone().cpu()
            for k, v in self.algo.model.state_dict().items()
        }
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algo.model.load_state_dict(
            torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")
