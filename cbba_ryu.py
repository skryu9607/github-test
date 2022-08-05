from dataclasses import field
import numpy as np
import time
import sys
import json
import matplotlib as plt
from dataclasses import dataclass, field
from task import Task
from agent import Agent
from World import World
@dataclass
class CBBA(object):
    num_agents: int  # number of agents
    num_tasks: int  # number of tasks
    max_depth: int  # maximum bundle depth
    # Time window is an interval in time during which an activity can or must take place.
    time_window_flag: bool  # True if time window exists
    duration_flag: bool  # Ture when all task duration > 0
    agent_types: list
    task_types: list
    space_limit_x: list  # [min, max] x coordinate [meter]
    space_limit_y: list  # [min, max] y coordinate [meter]
    space_limit_z: list  # [min, max] z coordinate [meter]
    time_interval_list: list  # time interval for all the agents and tasks
    agent_index_list: list  # 1D list
    bundle_list: list  # 2D list
    path_list: list  # 2D list
    times_list: list  # 2D list
    scores_list: list  # 2D list
    bid_list: list  # 2D list
    winners_list: list  # 2D list
    winner_bid_list: list  # 2D list
    graph: list  # 2D list represents the structure of graph
    Agentlist: list
    Tasklist: list
    World : World
    def __init__(self, config_data: dict):
        #list agent types
        '''
        Constructor 
        Initialize CBBA parameters 
        '''
        config_file_path = r'C:\Users\user\PycharmProjects\CBBA\config.json'
        with open(config_file_path,'r') as fp:
            config_data = json.load(fp)
        
        self.agent_types = config_data["AGENT_TYPES"]
        self.task_types =  config_data["TASK_TYPES"]
        self.compatibility_mat = [[0] * len(self.task_types) for _ in range(len(self.agent_types))]
        self.compatibility_mat[self.agent_types.index("quad")][self.task_types.indxe("track")] = 1
        self.compatibility_mat[self.agent_types.index("car")][self.task_types.indxe("rescue")] = 1
        self.time_interval_list = [min(int(config_data["TRACK_DEFAULT"]["START_TIME"]),
                                       int(config_data["RESCUE_DEFAULT"]["START_TIME"])),
                                   max(int(config_data["TRACK_DEFAULT"]["END_TIME"]),
                                       int(config_data["RESCUE_DEFAULT"]["END_TIME"]))]
        self.duration_flag = (min(int(config_data["TRACK_DEFAULT"]["DURATION"]),
                                  int(config_data["RESCUE_DEFAULT"]["DURATION"])) > 0)
    # agentlist 와 tasklist, world information을 받아서 구성하고, max_depth는 논문에서의 L_t, time_window_flag는 time_window가 없을때 false를 하여 feasibility 판별
    def settings(self,Agentlist: list, Tasklist:list, World:World , max_depth:int, time_window_flag:bool ):
        '''
            Initialize some lists given new Agentlist, Tasklist, and WorldInfoInput. 
        '''
        self.num_agents = len(Agentlist)
        self.num_tasks = len(Tasklist)
        self.max_depth = max_depth
        self.time_window_flag = time_window_flag

        self.Agentlist = Agentlist
        self.Tasklist = Tasklist

        self.World = World
        self.space_limit_x = World.limit_x
        self.space_limit_y = World.limit_y
        self.space_limit_z = World.limit_z

        # Fully connected graph
        ''' In MATLAB
        Graph = ~eye(N); % [0 1 1 1 1;
                   1 0 1 1 1;...;]
        '''
        M = np.ones(self.num_agents)
        MM = np.identity(self.num_agents)
        self.Graph = M-MM
        # -1 is the default value
        self.bundle_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.path_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.times_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.scores_list = [[-1] * self.max_depth for _ in range(self.num_agents)]

        # fixed the initialization, from 0 vector to -1 vector
        self.bid_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.winners_list = [[-1] * self.num_tasks for _ in range(self.num_agents)] # In paper this list is y 
        self.winner_bid_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.agent_index_list = []
        for n in range(self.num_agents):
            self.agent_index_list.append(self.Agentlist[n].agent_id)

    def bundle(self, idx_agent:int):
        self.bundle_remove(idx_agent)
        new_bid_flag = self.bundle_add(idx_agent)

        return new_bid_flag
    def bundle_add(self, idx_agent:int):






if __name__ == '__main__':
    config_file_path = r'C:\Users\user\PycharmProjects\CBBA\config.json'
    with open(config_file_path,'r') as fp:
        config_data = json.load(fp)
        print(json.dumps(config_data, indent = 4))
