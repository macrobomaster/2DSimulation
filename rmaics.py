# -*- coding: utf-8 -*-
# RoboMaster AI Challenge Simulator (RMAICS)
import math
from kernal import kernal
import random   
class rmaics(object):

    def __init__(self, agent_num, render=True):
        self.game = kernal(car_num=agent_num, render=render)
        self.g_map = self.game.get_map()
        self.memory = []
        self.prev_x = 0
        self.prev_y = 0
        self.prev_collision = 0

    def reset(self):
        self.state = self.game.reset()
        # state, object
        self.obs = self.get_observation(self.state)
        return self.obs

    def step(self, actions):
        state = self.game.step(actions)
        obs = self.get_observation(state)
        rewards = self.get_reward(state)

        self.memory.append([self.obs, actions, rewards])
        self.state = state

        return obs, rewards, state.done, None
    
    def get_observation(self, state):
        # print(state.agents)

        observation = {
            'time_remaining': state.time,
            'agent_states': state.agents,
            'competition_info': state.compet,
            'is_done': state.done,
            'visible_agents': state.detect,
            'visible_enemies': state.vision
        }
        return observation

    def euclidean_distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def normalize(self, value):
        return (value ) 

    def get_reward(self, state):
        reward = 0  # Initialize reward

        # Penalize collisions
        for agent in state.agents:
            temp_collision = agent[12] + agent[13] 

            if abs(self.prev_collision - temp_collision) > 0:
                reward -= 10  # Penalty for collisions
                print("Collision")
            self.prev_collision = temp_collision
        
        position_x = self.game.cars[0][1]
        position_y = self.game.cars[0][2]
        if self.prev_x != 0 and self.prev_y != 0:
            reward += self.euclidean_distance(position_x, position_y, self.prev_x, self.prev_y)
        self.prev_x = position_x
        self.prev_y = position_y

        # Reward for reaching a bonus area
        if state.compet[0][0] == 1 or state.compet[1][0] == 1:
            reward += 10  

        
        reward -= 0.05 
        
        return reward

    def play(self):
        self.game.play()

    def save_record(self, file):
        self.game.save_record(file)