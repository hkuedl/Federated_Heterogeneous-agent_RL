import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from Algorithm.model_filter import Filter

class Simulator:
    def __init__(self, num_iterations, agent, master, env, env_master, output, master_output, warmup, max_episode_length=None, variant = None,  **env_args):
        self.num_iterations = num_iterations
        self.agents = agent
        self.envs = env
        self.envs_master = env_master
        self.masters = master
        self.output = output
        self.master_output = master_output
        self.warmup = warmup
        self.max_episode_length = max_episode_length
        self.num_agents = len(self.agents)
        self.knowledge_distillation = False
        self.variant = variant
        self.converge_step = None


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

        #  Check the algorithm of master
        self.master_his_OBS = {}
        self.master_his_ACTION = {}
        self.master_his_TRADE = {}
        self.master_his_COST = {}
        self.master_his_COST_TUPLE = {}
        self.master_his_REWARD = {}
        self.master_his_OPERATION_COST = {}
        self.master_his_CARBON_COST = {}

        self.master_reward_curve = {}
        self.master_operation_cost_curve = {}
        self.master_carbon_cost_curve = {}
        self.master_cost_curve = {}

        for key, master in self.masters.items():
            self.master_his_OBS['agent{}'.format(key)] = []
            self.master_his_ACTION['agent{}'.format(key)] = []
            self.master_his_TRADE['agent{}'.format(key)] = []
            self.master_his_COST['agent{}'.format(key)] = []
            self.master_his_COST_TUPLE['agent{}'.format(key)] = []
            self.master_his_REWARD['agent{}'.format(key)] = []
            self.master_his_OPERATION_COST['agent{}'.format(key)] = []
            self.master_his_CARBON_COST['agent{}'.format(key)] = []
            self.master_reward_curve['agent{}'.format(key)] = []
            self.master_operation_cost_curve['agent{}'.format(key)] = []
            self.master_carbon_cost_curve['agent{}'.format(key)] = []
            self.master_cost_curve['agent{}'.format(key)] = []

    def train(self):
        episode = {}
        episode_steps = {}
        episode_reward = {}
        episode_operation_cost = {}
        episode_carbon_cost = {}
        episode_cost = {}
        observation = {}
        agent_step = 0
        for key, agent in self.agents.items():
            agent.is_training = True
            episode[key] = 0
            episode_steps[key] = 0
            episode_reward[key] = 0.
            episode_operation_cost[key] = 0.
            episode_carbon_cost[key] = 0.
            episode_cost[key] = 0.
            observation[key] = None

        master_episode = {}
        master_episode_steps = {}
        master_episode_reward = {}
        master_episode_operation_cost = {}
        master_episode_carbon_cost = {}
        master_episode_cost = {}
        master_observation = {}
        
        for key, master in self.masters.items():
            master.is_training = True
            master_episode[key] = 0
            master_episode_steps[key] = 0
            master_episode_reward[key] = 0.
            master_episode_operation_cost[key] = 0.
            master_episode_carbon_cost[key] = 0.
            master_episode_cost[key] = 0.
            master_observation[key] = None

        # broadcast the initial master parameters to all agents
        for key, master in self.masters.items():
            if key == 1:
                actor_state_dict = master.actor.state_dict()
                actor_target_state_dict = master.actor_target.state_dict()
                critic_state_dict = master.critic.state_dict()
                critic_target_state_dict = master.critic_target.state_dict()
                guard_state_dict = master.guard.state_dict()
                guard_target_state_dict = master.guard_target.state_dict()
            else:
                master.actor.load_state_dict(actor_state_dict)
                master.actor_target.load_state_dict(actor_target_state_dict)
                master.critic.load_state_dict(critic_state_dict)
                master.critic_target.load_state_dict(critic_target_state_dict)
                master.guard.load_state_dict(guard_state_dict)
                master.guard_target.load_state_dict(guard_target_state_dict)

        global_model = deepcopy(self.masters[1])

        for step in tqdm(range(self.num_iterations)):
            # maseter training
            for key, master in self.masters.items():
                # reset if it is the start of episode
                if master_observation[key] is None:
                    _, master_observation[key], done = deepcopy(self.envs_master[key].reset_SOC(master_episode[key]))  
                    master.reset(master_observation[key])

                # master pick action ...
                if step <= self.warmup:
                    action_raw = master.random_action()
                else:
                    action_raw = master.select_action(master_observation[key])

                # action correction and cost calculation
                action, trade, cost_tuple = self.envs_master[key].cost_calculation(master_episode[key], master_episode_steps[key], master_observation[key], action_raw)
                cost = -np.array(cost_tuple).sum()

                # env response with next_observation, reward, operation cost, and terminal flag 
                _, master_observation2, reward, oc_cost, done = self.envs_master[key].step(master_episode[key], action, trade, master_episode_steps[key], self.max_episode_length)
                master_observation2 = deepcopy(master_observation2)
                if self.max_episode_length and master_episode_steps[key] >= self.max_episode_length -1:
                    done = True

                # master observe and update policy
                master.observe(reward, cost, master_observation2, done)
                if step > self.warmup:
                    master.update_policy()
                
                # update                                                                        
                master_episode_steps[key] += 1
                master_episode_reward[key] += reward
                master_episode_operation_cost[key] += oc_cost[0]
                master_episode_carbon_cost[key] += oc_cost[1]
                master_episode_cost[key] += cost

                # write log
                self.master_logger(master_observation[key], action, trade, cost, reward, cost_tuple, oc_cost, key)
                master_observation[key] = deepcopy(master_observation2)

                if done: # end of episode
                    master.memory.append(
                        master_observation[key],
                        master.select_action(master_observation[key]),
                        0., 0., False
                    )

                    # reset
                    master_observation[key] = None
                    master_episode_steps[key] = 0
                    self.master_reward_curve['agent{}'.format(key)].append(master_episode_reward[key])
                    self.master_operation_cost_curve['agent{}'.format(key)].append(master_episode_operation_cost[key])
                    self.master_carbon_cost_curve['agent{}'.format(key)].append(master_episode_carbon_cost[key])
                    self.master_cost_curve['agent{}'.format(key)].append(master_episode_cost[key])
                    master_episode_reward[key] = 0.
                    master_episode_operation_cost[key] = 0.
                    master_episode_carbon_cost[key] = 0.
                    master_episode_cost[key] = 0.
                    master_episode[key] += 1

            if (step > self.warmup) and (step % 1000 == 0) and (step < self.variant['master_stop'] * self.max_episode_length + 1):
                local_update_critic = []
                local_update_guard = []
                local_update_actor = []
                local_update_critic_target = []
                local_update_guard_target = []
                local_update_actor_target = []
                
                for key, master in self.masters.items():
                    model_update_critic = {k: master.critic.state_dict()[k] - global_model.critic.state_dict()[k] for k in global_model.critic.state_dict().keys()}
                    local_update_critic.append(model_update_critic)
                    model_update_guard = {k: master.guard.state_dict()[k] - global_model.guard.state_dict()[k] for k in global_model.guard.state_dict().keys()}
                    local_update_guard.append(model_update_guard)
                    model_update_actor = {k: master.actor.state_dict()[k] - global_model.actor.state_dict()[k] for k in global_model.actor.state_dict().keys()}
                    local_update_actor.append(model_update_actor)
                    model_update_critic_target = {k: master.critic_target.state_dict()[k] - global_model.critic_target.state_dict()[k] for k in global_model.critic_target.state_dict().keys()}
                    local_update_critic_target.append(model_update_critic_target)
                    model_update_guard_target = {k: master.guard_target.state_dict()[k] - global_model.guard_target.state_dict()[k] for k in global_model.guard_target.state_dict().keys()}
                    local_update_guard_target.append(model_update_guard_target)
                    model_update_actor_target = {k: master.actor_target.state_dict()[k] - global_model.actor_target.state_dict()[k] for k in global_model.actor_target.state_dict().keys()}
                    local_update_actor_target.append(model_update_actor_target)

                # filtering model update
                global_critic, _ = Filter(local_update_critic, global_model.critic.state_dict(), self.variant)
                global_actor, _ = Filter(local_update_actor, global_model.actor.state_dict(), self.variant)
                global_critic_target, _ = Filter(local_update_critic_target, global_model.critic_target.state_dict(), self.variant)
                global_actor_target, _ = Filter(local_update_actor_target, global_model.actor_target.state_dict(), self.variant)
                global_guard, _ = Filter(local_update_guard, global_model.guard.state_dict(), self.variant)
                global_guard_target, _ = Filter(local_update_guard_target, global_model.guard_target.state_dict(), self.variant)

                global_model.critic.load_state_dict(global_critic)
                global_model.guard.load_state_dict(global_guard)
                global_model.actor.load_state_dict(global_actor)
                global_model.critic_target.load_state_dict(global_critic_target)
                global_model.guard_target.load_state_dict(global_guard_target)
                global_model.actor_target.load_state_dict(global_actor_target)

                # broadcast the master parameters to all agents
                for key, master in self.masters.items():
                    master.actor.load_state_dict(global_model.actor.state_dict())
                    master.actor_target.load_state_dict(global_model.actor_target.state_dict())
                    master.critic.load_state_dict(global_model.critic.state_dict())
                    master.critic_target.load_state_dict(global_model.critic_target.state_dict())
                    master.guard.load_state_dict(global_model.guard.state_dict())
                    master.guard_target.load_state_dict(global_model.guard_target.state_dict())

                if step == self.variant['master_stop'] * self.max_episode_length:
                    self.masters[1].save_model(self.master_output, '_MF_FRL')
            
            # save the fine-tuned master model for each agent
            if step == (11000*self.max_episode_length-1):
                for key, master in self.masters.items():
                    master.save_model(self.master_output, '_FT1000_'+str(key))
    
        self.save_logger()

    def master_logger(self, observation, action, trade, cost, reward, cost_tuple, oc_cost, key):
        self.master_his_OBS['agent{}'.format(key)].append(observation)
        self.master_his_ACTION['agent{}'.format(key)].append(action)
        self.master_his_TRADE['agent{}'.format(key)].append(trade)
        self.master_his_COST['agent{}'.format(key)].append(cost)
        self.master_his_COST_TUPLE['agent{}'.format(key)].append(cost_tuple)
        self.master_his_REWARD['agent{}'.format(key)].append(reward)
        self.master_his_OPERATION_COST['agent{}'.format(key)].append(oc_cost[0])      
        self.master_his_CARBON_COST['agent{}'.format(key)].append(oc_cost[1])                                                      

        return None

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
        for key, master in self.masters.items():
            np.save(self.master_output+'/his_obs_agent{}.npy'.format(key), np.array(self.master_his_OBS['agent{}'.format(key)]))
            np.save(self.master_output+'/his_action_agent{}.npy'.format(key), np.array(self.master_his_ACTION['agent{}'.format(key)]))
            np.save(self.master_output+'/his_trade_agent{}.npy'.format(key), np.array(self.master_his_TRADE['agent{}'.format(key)]))
            np.save(self.master_output+'/his_cost_agent{}.npy'.format(key), np.array(self.master_his_COST['agent{}'.format(key)]))
            np.save(self.master_output+'/his_reward_agent{}.npy'.format(key), np.array(self.master_his_REWARD['agent{}'.format(key)]))
            np.save(self.master_output+'/his_operation_cost_agent{}.npy'.format(key), np.array(self.master_his_OPERATION_COST['agent{}'.format(key)]))
            np.save(self.master_output+'/his_carbon_cost_agent{}.npy'.format(key), np.array(self.master_his_CARBON_COST['agent{}'.format(key)]))
            np.save(self.master_output+'/Episode_reward_agent{}.npy'.format(key), np.array(self.master_reward_curve['agent{}'.format(key)]))
            np.save(self.master_output+'/Episode_operation_cost_agent{}.npy'.format(key), np.array(self.master_operation_cost_curve['agent{}'.format(key)]))
            np.save(self.master_output+'/Episode_carbon_cost_agent{}.npy'.format(key), np.array(self.master_carbon_cost_curve['agent{}'.format(key)]))
            np.save(self.master_output+'/Episode_cost_agent{}.npy'.format(key), np.array(self.master_cost_curve['agent{}'.format(key)]))
            np.save(self.master_output+'/his_cost_tuple_agent{}.npy'.format(key), np.array(self.master_his_COST_TUPLE['agent{}'.format(key)]))
                        
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



