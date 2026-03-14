import numpy as np
from copy import deepcopy
import torch


from Env.EH_model import EH_Model
from Algorithm.agent import RL_agent
from Algorithm.master import RL_master
from Train.simulator_proposed2 import Simulator1
from Train.simulator_proposed2_KD import Simulator2
from Algorithm.util import *


def global_model_training(variant):
    device = torch.device(variant['device'])

    master = {}
    agent = {}
    env = {}
    env_master = {}
    for i in range(variant['num_agent']):
        key = i + 1
        if variant['agent_preference'] is None:
            env[key] =  EH_Model(0.5, variant['agent_config'][i], 1, index=variant['dataset_index'][i])
            env_master[key] =  EH_Model(0.5, variant['agent_config'][i], 1, index=variant['dataset_index'][i])
        elif variant['agent_config'] is None:
            env[key] =  EH_Model(variant['agent_preference'][i], [1,1,1], variant['reward_scale'][i], index=variant['dataset_index'][i])
            env_master[key] =  EH_Model(variant['agent_preference'][i], [1,1,1], variant['reward_scale'][i], index=variant['dataset_index'][i])
        else:
            env[key] =  EH_Model(variant['agent_preference'][i], variant['agent_config'][i], variant['reward_scale'][i], index=variant['dataset_index'][i])
            env_master[key] =  EH_Model(variant['agent_preference'][i], variant['agent_config'][i], variant['reward_scale'][i], index=variant['dataset_index'][i])
        obs_dim = env[key].get_obs_dim()
        action_dim = env[key].get_action_dim()
        agent[key] = RL_agent(obs_dim, action_dim, variant, device, variant['seed'][i])
        master[key] = RL_master(obs_dim, action_dim, variant, device, variant['seed'][i])

    simulator = Simulator1(variant['train_iter'], agent, master, env, env_master, variant['output'], variant['master_output'], variant['warmup'], variant['max_episode_length'], variant)

    simulator.train()

def know_transfer(variant):
    device = torch.device(variant['device'])

    master = {}
    agent = {}
    env = {}
    env_master = {}
    for i in range(variant['num_agent']):
        key = i + 1
        if variant['agent_preference'] is None:
            env[key] =  EH_Model(0.5, variant['agent_config'][key-1], 1, scale=variant['dataset_scale'][key-1])
            env_master[key] =  EH_Model(0.5, variant['agent_config'][key-1], 1, scale=variant['dataset_scale'][key-1])
        elif variant['agent_config'] is None:
            env[key] =  EH_Model(variant['agent_preference'][key-1], [1,1,1], variant['reward_scale'][key-1], scale=variant['dataset_scale'][key-1])
            env_master[key] =  EH_Model(variant['agent_preference'][key-1], [1,1,1], variant['reward_scale'][key-1], scale=variant['dataset_scale'][key-1])
        else:
            env[key] =  EH_Model(variant['agent_preference'][key-1], variant['agent_config'][key-1], variant['reward_scale'][key-1], scale=variant['dataset_scale'][key-1])
            env_master[key] =  EH_Model(variant['agent_preference'][key-1], variant['agent_config'][key-1], variant['reward_scale'][key-1], scale=variant['dataset_scale'][key-1])
        obs_dim = env[key].get_obs_dim()
        action_dim = env[key].get_action_dim()
        agent[key] = RL_agent(obs_dim, action_dim, variant, device, variant['seed'][key-1])
        master[key] = RL_master(obs_dim, action_dim, variant, device, variant['seed'][key-1])

    simulator = Simulator2(variant['train_iter'], agent, master, env, env_master, variant['output'], variant['master_input'], variant['warmup'], variant['max_episode_length'], variant)

    simulator.train()


if __name__ == "__main__":

    variant1 = dict(
        algorithm = "Safety_aware RL",
        mode = 'train',
        master_output = '/home/user/data/FHRL/Global_Model_Training/master',
        device = "cuda:0", 
        index = [0, 1, 4, 5],               # action范围在[0,1]之间的action index, e.g., dg, gt, ehp, gb.

        train_iter = 264000,                # total number of iterations for training
        master_stop = 10000,                # after a certain number of episodes, the master stops training
        warmup = 3000,                      # time without training but only filling the replay memory  (334days * 24h/day)
        max_episode_length = 24,            # maximum length of an episode (24h a day)

        rate = 0.001,                       # learning rate  
        prate = 0.0001,                     # policy net learning rate    
        bsize = 256,                        # mini-batch size of agents
        mbsize = 256,                       # mini-batch size of masters
        rmsize = 1000000,                   # memory size                 
        epsilon = 180000,                   # linear decay of exploration policy 
        hidden1 = 256,                      # hidden num of first fully connect layer                                     
        hidden2 = 256,                      # hidden num of second fully connect layer                      
        discount = 0.999,                   # reward discount factor
        ou_theta = 0.15,                    # OU noise parameter θ
        ou_sigma = 0.4,                     # OU noise parameter σ
        ou_mu = 0.0,                        # OU noise parameter μ
        init_w = 0.003,                     # initial policy net weight
        window_length = 1,                  # the number of memories to be used for training
        tau = 0.001,                        # target smoothing coefficient(τ) 
        
        ## 10 agents
        num_agent = 10,
        seed = [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,],                          # random seed

        dataset_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

        agent_preference = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0],

        reward_scale = [1, 1, 1, 1, 1.5, 2, 3, 6, 10, 10],

        agent_config = [[1,1,1], [0.9,0.9,0.9], [0.91,0.91,0], [0.88,0.88,0.88], [0.95,0.95,0.95], 
                        [0,0.97,0.97], [0.93,0.93,0.93], [0.92,0,0.92], [0.96,0.96,0.96], [0,0,0]],
        )

    global_model_training(variant1)

    variant2 = dict(
        algorithm = "Safety-aware RL",
        mode = 'train',
        master_input = '/home/user/data/FHRL/Global_Model_Training/master',
        output = '/home/user/data/FHRL/Knowledge_Transfer/Agent',     
        
        device = "cuda:0", 
        index = [0, 1, 4, 5],               # action范围在[0,1]之间的action index, e.g., dg, gt, ehp, gb.

        train_iter = 264000,                # total number of iterations for training
        warmup = 3000,                      # time without training but only filling the replay memory  (334days * 24h/day)
        max_episode_length = 24,            # maximum length of an episode (24h a day) 

        rate = 0.001,                       # learning rate  
        prate = 0.0001,                     # policy net learning rate    
        bsize = 256,                        # mini-batch size of agents
        mbsize = 256,                       # mini-batch size of masters
        rmsize = 1000000,                   # memory size                
        epsilon = 180000,                   # linear decay of exploration policy 
        hidden1 = 256,                      # hidden num of first fully connect layer                                     
        hidden2 = 256,                      # hidden num of second fully connect layer                      
        discount = 0.999,                   # reward discount factor
        ou_theta = 0.15,                    # OU noise parameter θ
        ou_sigma = 0.4,                     # OU noise parameter σ
        ou_mu = 0.0,                        # OU noise parameter μ
        init_w = 0.003,                     # initial policy net weight
        window_length = 1,                  # the number of memories to be used for training
        tau = 0.001,                        # target smoothing coefficient(τ) 
        
        # agent setting
        ## 10 agents
        agent_train = True,
        KD_start = 0,                      # the iteration to start KD
        KD_stop = 10000*24,                # the iteration to stop KD

        ## 10 agents
        num_agent = 10,
        seed = [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,],                          # random seed

        dataset_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

        agent_preference = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0],

        reward_scale = [1, 1, 1, 1, 1.5, 2, 3, 6, 10, 10],

        agent_config = [[1,1,1], [0.9,0.9,0.9], [0.91,0.91,0], [0.88,0.88,0.88], [0.95,0.95,0.95], 
                        [0,0.97,0.97], [0.93,0.93,0.93], [0.92,0,0.92], [0.96,0.96,0.96], [0,0,0]],
        )
    
    know_transfer(variant2)









