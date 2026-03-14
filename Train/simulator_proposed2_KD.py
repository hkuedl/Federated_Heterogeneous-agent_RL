import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from Algorithm.model_filter import Filter


class Simulator:
    def __init__(self, num_iterations, agent, master, env, env_master, output, master_input, warmup, max_episode_length=None, variant = None,  **env_args):
        self.num_iterations = num_iterations
        self.agents = agent
        self.envs = env
        self.envs_master = env_master
        self.masters = master
        self.output = output
        self.master_input = master_input
        self.warmup = warmup
        self.max_episode_length = max_episode_length
        self.num_agents = len(self.agents)
        self.knowledge_distillation = False
        self.variant = variant

        #  Check the algorithm
        self.his_OBS = {}
        self.his_ACTION = {}
        self.his_TRADE = {}
        self.his_COST = {}
        self.his_COST_TUPLE = {}
        self.his_REWARD = {}
        self.his_OPERATION_COST = {}
        self.his_CARBON_COST = {}

        self.reward_curve = {}
        self.operation_cost_curve = {}
        self.carbon_cost_curve = {}
        self.cost_curve = {}

        for key, agent in self.agents.items():
            self.his_OBS['agent{}'.format(key)] = []
            self.his_ACTION['agent{}'.format(key)] = []
            self.his_TRADE['agent{}'.format(key)] = []
            self.his_COST['agent{}'.format(key)] = []
            self.his_COST_TUPLE['agent{}'.format(key)] = []
            self.his_REWARD['agent{}'.format(key)] = []
            self.his_OPERATION_COST['agent{}'.format(key)] = []
            self.his_CARBON_COST['agent{}'.format(key)] = []
            self.reward_curve['agent{}'.format(key)] = []
            self.operation_cost_curve['agent{}'.format(key)] = []
            self.carbon_cost_curve['agent{}'.format(key)] = []
            self.cost_curve['agent{}'.format(key)] = []

    def train(self):
        episode = {}
        episode_steps = {}
        episode_reward = {}
        episode_operation_cost = {}
        episode_carbon_cost = {}
        episode_cost = {}
        observation = {}
        for key, agent in self.agents.items():
            agent.is_training = True
            episode[key] = 0
            episode_steps[key] = 0
            episode_reward[key] = 0.
            episode_operation_cost[key] = 0.
            episode_carbon_cost[key] = 0.
            episode_cost[key] = 0.
            observation[key] = None

        # load master model
        for key, master in self.masters.items():
            actor_address = self.master_input+'actor_MF_FRL_FT1000_{}.pkl'.format(key)
            critic_address = self.master_input+'critic_MF_FRL_FT1000_{}.pkl'.format(key)
            guard_address = self.master_input+'guard_MF_FRL_FT1000_{}.pkl'.format(key)
            master.actor.load_state_dict(torch.load(actor_address))
            master.critic.load_state_dict(torch.load(critic_address))
            master.guard.load_state_dict(torch.load(guard_address))

        # agent training
        for step in tqdm(range(self.num_iterations)):
            if (step > self.variant['KD_start']-1) and (step < self.variant['KD_stop']):
                self.knowledge_distillation = True
            else:
                self.knowledge_distillation = False

            for key, agent in self.agents.items():
                # reset if it is the start of episode
                if observation[key] is None:
                    _, observation[key], done = deepcopy(self.envs[key].reset_SOC(episode[key]))  
                    agent.reset(observation[key])

                # agent pick action ...
                if step <= self.warmup:
                    action_raw = agent.random_action()
                else:
                    action_raw = agent.select_action(observation[key])

                # action correction and cost calculation
                action, trade, cost_tuple = self.envs[key].cost_calculation(episode[key], episode_steps[key], observation[key], action_raw)
                cost = -np.array(cost_tuple).sum()

                # env response with next_observation, reward, operation cost, and terminal flag 
                _, observation2, reward, oc_cost, done = self.envs[key].step(episode[key], action, trade, episode_steps[key], self.max_episode_length)
                observation2 = deepcopy(observation2)
                if self.max_episode_length and episode_steps[key] >= self.max_episode_length -1:
                    done = True

                # agent observe and update policy
                agent.observe(reward, cost, observation2, done)
                if step > self.warmup:
                    if self.knowledge_distillation:
                        alpha_KD = (step/self.num_iterations)                                    
                        # alpha_KD = 0.5                                                         
                        agent.update_policy_KD(self.masters[key], alpha_KD)                      
                    else:
                        agent.update_policy()                                                    

                # update 
                episode_steps[key] += 1
                episode_reward[key] += reward
                episode_operation_cost[key] += oc_cost[0]
                episode_carbon_cost[key] += oc_cost[1]
                episode_cost[key] += cost

                # write log
                self.logger(observation[key], action, trade, cost, reward, cost_tuple, oc_cost, key)
                observation[key] = deepcopy(observation2)

                if done: # end of episode
                    agent.memory.append(
                        observation[key],
                        agent.select_action(observation[key]),
                        0., 0., False
                    )

                    # reset
                    observation[key] = None
                    episode_steps[key] = 0
                    self.reward_curve['agent{}'.format(key)].append(episode_reward[key])
                    self.operation_cost_curve['agent{}'.format(key)].append(episode_operation_cost[key])
                    self.carbon_cost_curve['agent{}'.format(key)].append(episode_carbon_cost[key])
                    self.cost_curve['agent{}'.format(key)].append(episode_cost[key])
                    episode_reward[key] = 0.
                    episode_operation_cost[key] = 0.
                    episode_carbon_cost[key] = 0.
                    episode_cost[key] = 0.
                    episode[key] += 1
                    
                if (step > 0) and step % 5000 == 0:
                    np.save(self.output+str(key)+'/Episode_reward_agent{}.npy'.format(key), np.array(self.reward_curve['agent{}'.format(key)]))
                    np.save(self.output+str(key)+'/Episode_operation_cost_agent{}.npy'.format(key), np.array(self.operation_cost_curve['agent{}'.format(key)]))
                    np.save(self.output+str(key)+'/Episode_carbon_cost_agent{}.npy'.format(key), np.array(self.carbon_cost_curve['agent{}'.format(key)]))
                    np.save(self.output+str(key)+'/Episode_cost_agent{}.npy'.format(key), np.array(self.cost_curve['agent{}'.format(key)]))


        self.save_logger()
        for key, agent in self.agents.items():
            agent.save_model(self.output+str(key), '_MF_FRL_FT_agent_' + str(key))


    def logger(self, observation, action, trade, cost, reward, cost_tuple, oc_cost, key):
        self.his_OBS['agent{}'.format(key)].append(observation)
        self.his_ACTION['agent{}'.format(key)].append(action)
        self.his_TRADE['agent{}'.format(key)].append(trade)
        self.his_COST['agent{}'.format(key)].append(cost)
        self.his_COST_TUPLE['agent{}'.format(key)].append(cost_tuple)
        self.his_REWARD['agent{}'.format(key)].append(reward)
        self.his_OPERATION_COST['agent{}'.format(key)].append(oc_cost[0])   
        self.his_CARBON_COST['agent{}'.format(key)].append(oc_cost[1])                                                         

        return None

    def save_logger(self):
        for key, agent in self.agents.items():
            np.save(self.output+str(key)+'/his_obs_agent{}.npy'.format(key), np.array(self.his_OBS['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_action_agent{}.npy'.format(key), np.array(self.his_ACTION['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_trade_agent{}.npy'.format(key), np.array(self.his_TRADE['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_cost_agent{}.npy'.format(key), np.array(self.his_COST['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_reward_agent{}.npy'.format(key), np.array(self.his_REWARD['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_operation_cost_agent{}.npy'.format(key), np.array(self.his_OPERATION_COST['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_carbon_cost_agent{}.npy'.format(key), np.array(self.his_CARBON_COST['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/Episode_reward_agent{}.npy'.format(key), np.array(self.reward_curve['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/Episode_operation_cost_agent{}.npy'.format(key), np.array(self.operation_cost_curve['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/Episode_carbon_cost_agent{}.npy'.format(key), np.array(self.carbon_cost_curve['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/Episode_cost_agent{}.npy'.format(key), np.array(self.cost_curve['agent{}'.format(key)]))
            np.save(self.output+str(key)+'/his_cost_tuple_agent{}.npy'.format(key), np.array(self.his_COST_TUPLE['agent{}'.format(key)]))
                
        return  None
    
    def moving_average(self, array, window):
    
        MA = []
        for i in range(array.shape[0] - window + 1):
            front = i
            tail = i + window
            MA.append(array[front : tail].mean())

        return np.array(MA)
    
    def padded_list(self, input_list, max_len):
        max_length = max_len
        padded_list = [sublist + [np.nan] * (max_length - len(sublist)) for sublist in input_list]

        numpy_array = np.array(padded_list)

        return numpy_array



