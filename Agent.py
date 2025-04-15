import time, enum, math
import numpy as np
import pandas as pd
import pylab as plt
import sys
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

import networkx as nx
# from numba import jit

from state import State


class MyAgent(Agent):
    
    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model, se_flag, se_threshold, topic_weights, G, seed):
        super().__init__(unique_id, model)
        
        self.sim_model = model
        self.user_id = unique_id
        self.state = State.NOT_RECEIVED
        self.se_flag = se_flag
        self.G = G
        self.se_threshold = se_threshold
        self.topic_weights = topic_weights
        self.sign = lambda x: x and (1, -1)[x<0]
        
#         self.seed = seed
#         print(type(self.sim_model))
#         self.sim_model.reset_randomizer(seed)
        random.seed(seed)
        
    def create_post(self):
        
        self.state = State.ORIGIN
        self.model.schedule.add(self)
        # self.sim_model.G_share.add_node(self.user_id)
        
    def get_user_details(self, user_id, topic_id):
        
        activity = self.G.nodes[user_id]['activity']
        stance = self.G.nodes[user_id]['topic_' + str(int(topic_id))]
        privacy = self.G.nodes[user_id]['privacy']
        
        return activity, stance, privacy
        
    def compute_sanction_score(self, user_id):
        
        author = self.sim_model.post['author']
        topic_id = self.sim_model.post['topic']
        post_stance = self.sim_model.post['stance']
        
        activity, stance, privacy = self.get_user_details(user_id, topic_id)
        
        diff = abs(stance - post_stance)
        # dir_mov = self.sign(diff)
        # dir_mov = abs(diff)
        # print("activity ", activity)
        # print("stance ", stance)
        # print(" post_stance ", post_stance)
        
        sanction_score = stance * post_stance * 10 * self.topic_weights[topic_id]
        
        # print('user details ', user_id, activity, stance, privacy, topic_id, post_stance, self.topic_weights[topic_id], sanction_score)
        
        # sys.exit()
        
        # if sanction_score == 0:
        #     print('author ', author)
        #     print('topic_id ', topic_id)
        #     print('post_stance ', post_stance)
        #     print('activity ', activity)
        #     print('stance ', stance)
#         sanction_score = dir_mov * (activity * self.topic_weights[topic_id])/(diff*diff + 1) 
        #print("activity ", author, user_id, post_stance, activity, sanction_score)
            
        return round(sanction_score, 6)
    
    def sigmoid_weighted_product(self, a, b, c, d, k=5):
        
        prob = (a * b * abs(c * d)) / (1 + np.exp(-k * (c * d - abs(c - d))))
        return max(0.25, min(1, prob))
    
    def compute_sharing_probability(self, user_id):
        
        topic_id = self.sim_model.post['topic']
        post_stance = self.sim_model.post['stance']
        
        activity, stance, privacy = self.get_user_details(user_id, topic_id)
        
        diff = abs(stance - post_stance)
        # dir_mov = abs(diff)
#         sharing_prob = (activity * privacy * self.topic_weights[topic_id])/(diff*diff + 1)
        # sharing_prob = activity * abs(stance * post_stance) * privacy * 10 * self.topic_weights[topic_id]
        # sharing_prob = activity * self.sigmoid_weighted_product(stance, post_stance) * privacy * 10 * self.topic_weights[topic_id]
        
        sharing_prob = self.sigmoid_weighted_product(activity, privacy, stance, post_stance)
        # sharing_prob = 1
           
        return round(sharing_prob, 6)
        
    def current_status(self):
        """Check current status"""
            
        if self.state == State.ORIGIN:
            self.share_post()
        
        elif self.state == State.RECEIVED:
            
            sharing_prob = self.compute_sharing_probability(self.user_id)
            rnd = random.random()
            
            if(rnd < sharing_prob):
                
                self.share_post()
            
            else:
                self.state = State.DISINTERESTED
                
        elif self.state == State.DISINTERESTED:
            self.model.schedule.remove(self)
            

    def receiving_agents(self, agent):
        
        uid = agent.user_id
        
        # if self.user_id not in [100, 101, 102]:
        agent.state = State.RECEIVED
        self.model.schedule.add(agent)

        sanction_score = self.compute_sanction_score(uid)
        
        if sanction_score != 0:
            self.sim_model.G_share.add_node(uid)
            self.sim_model.G_share.add_edge(uid, self.user_id, weight = sanction_score)
            
        # if self.user_id == 3:
        #     print("sanction_score 1 ", sanction_score, uid)
            # print(self.sim_model.G_share.in_degree(uid))
            
        # if uid == 1:
        #     print("sanction_score 2 ", sanction_score, self.user_id)
    

    def selective_exposure(self, neighbor_nodes):
        
        neighbor_nodes = list(neighbor_nodes)
        topic_id = self.sim_model.post['topic']
        
        stance = self.G.nodes[self.user_id]['topic_' + str(int(topic_id))]
        neighbour_stances = [self.G.nodes[_id]['topic_' + str(int(topic_id))] for _id in neighbor_nodes]
        
        nstance_df = pd.DataFrame()
        nstance_df['stance'] = [self.G.nodes[_id]['topic_' + str(int(topic_id))] for _id in neighbor_nodes]
        nstance_df['user_id'] = neighbor_nodes
        
        nstance_df =  nstance_df.assign(stance_diff = lambda x: (x['stance'] - stance))
        #nstance_df['stance'] - stance
        
        #df.assign(Percentage = lambda x: (x['Total_Marks'] /500 * 100))
        nstance_df['stance_diff'] = nstance_df['stance_diff'].abs()
        
        selected_neighbors = nstance_df[nstance_df['stance_diff'] <= self.se_threshold]['user_id']
        
#         pol_inclination = data[data['id'] == self.user_id]['topic_'+str(topic_id)].values[0]
#         neighbor_inclination = data[data['id'].isin(neighbor_nodes)]
        
#         diff = neighbor_inclination['topic_'+str(topic_id)] - pol_inclination
#         diff = diff.abs()
        
#         selected_neighbors = diff[diff <= se_threshold]
        
        return list(selected_neighbors)
    
    
    def share_post(self):
        """Find friends and share the post with them"""
        
        neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbor_nodes = [x for x in neighbor_nodes if x not in {100, 101, 102}]
        if(self.se_flag == True):
            selected_neighbors = self.selective_exposure(neighbor_nodes)
        else:
            selected_neighbors = neighbor_nodes
        
        neighbor_agents = [
            agent
            for agent in self.model.grid.get_cell_list_contents(selected_neighbors)
            if (agent.state == State.NOT_RECEIVED)
        ]
        
        if len(neighbor_agents) > 0:
            self.state = State.SPREADER
            
        else:
            self.state = State.RECEIVED
            
        for agent in neighbor_agents:
            self.receiving_agents(agent)
            
#         if self.user_id == 3:
            
#             print("neighbor_agents ", neighbor_agents)
            
        self.model.schedule.remove(self)
    
    
    def step(self):
        
        self.current_status()