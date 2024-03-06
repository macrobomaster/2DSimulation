# -*- coding: utf-8 -*-
# RoboMaster AI Challenge Simulator (RMAICS)

from kernal import kernal
import random   
class rmaics(object):

    def __init__(self, agent_num, render=True):
        self.game = kernal(car_num=agent_num, render=render)
        self.g_map = self.game.get_map()
        self.memory = []

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
        observation = {
            'time_remaining': state.time,
            'agent_states': state.agents,
            'competition_info': state.compet,
            'is_done': state.done,
            'visible_agents': state.detect,
            'visible_enemies': state.vision
        }
        return observation

    
    def get_reward(self, state):
        reward = 0  # Initialize reward

        # Penalize collisions
        for agent in state.agents:
            if agent[12] > 0 or agent[13] > 0 or agent[14] > 0:
                reward -= 1  # Penalty for collisions
        
        # Reward for reaching a bonus area
        if state.compet[0][0] == 1 or state.compet[1][0] == 1:
            reward += 10  

        
        reward -= 0.05 
        
        return reward

    def play(self):
        self.game.play()

    def save_record(self, file):
        self.game.save_record(file)