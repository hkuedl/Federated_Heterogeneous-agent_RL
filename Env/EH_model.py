import gym
from gym import spaces
import numpy as np
import pandas as pd
import math
import copy


class EH_Model(gym.Env):
    def __init__(self, preference = 0, config = [1, 1, 1], reward_scale=1, test = False, index = 1):
        
        self.oneday_flag = False
        self.config = config               # 1: es, 2: gs, 3: ts              
        self.test = test
        self.index = index

        ## Parameters setting
        self.gamma = 0.9                              # 用于计算reward
        self.alpha = preference                       # 减碳目标权重
        self.reward_scale = reward_scale              # reward缩放系数
        self.std = 0.2                                # 环境不确定性，e.g., demand, RES
        self.operator_name = ['pso', 'gso', 'hso']    #运营商名称
        self.obs_num = [7, 2, 3]                      #各运营商observation数量
        self.action_dim = 7
        self.obs_dim = 12
        self.seed = 0
        ### parameters of storages
        self.es_capacity = 200  * self.config[0]         # energy capacity of energy storage
        self.gs_capacity = 200   * self.config[1]         # energy capacity of gas storage 
        self.ts_capacity = 400  * self.config[2]         # energy capacity of thermal storage
        self.es_efficiency = 1                           # 充放效率, ref:0.95
        self.gs_efficiency = 1                           # 充放效率, ref:0.90
        self.ts_efficiency = 1                           # 充放效率, ref:0.90
        ### power max of devices 
        self.grid_max = 999                            # max power of grid import and export
        self.gas_max = 999                             # max power of gas grid import and export
        self.dg_power_max = 150                        # max power out of diesel generator
        self.gt_power_max = 150                        # max power out of gas turbine
        self.gb_power_max = 200                        # max power out of gas boiler
        self.ehp_power_max = 200                       # max power out of electric heat pump
        self.es_power_max = 50 * self.config[0]        # max power out of energy storage
        self.gs_power_max = 50 * self.config[1]        # max power out of gas storage 
        self.ts_power_max = 100 * self.config[2]        # max power out of heat storage
        ### production cost of devices
        self.cost_dg = 0.270  
        self.cost_gt = 0.150
        self.cost_gb = 0.05
        self.cost_ehp = 0.04
        ### carbon emission rate of devices
        self.cer_dg = 0.000600      # t/kWh
        self.cer_gt = 0.000368     # t/kWh
        self.cer_gb = 0.000234     # t/kWh
        self.cer_grid = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 6)  # dynamic carbon emission factor of the grid, t/kWh
        self.cer_grid = np.array(self.cer_grid).reshape((365,24))
        carbon_price = 50          # 50 pound / t

        ## Data loading
        energy_demand = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 0)
        heat_demand = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 1)
        gas_demand = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 2)
        elec_price = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 3)
        gas_price = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 4)
        res = pd.read_excel('/home/user/workspaces/FHRL/Env/dataset'+str(self.index)+'.xlsx', sheet_name = 5)

        ### Demand
        self.energy_demand = np.array(energy_demand).reshape((365,24,3))
        self.gas_demand = np.array(gas_demand)
        self.heat_demand = np.array(heat_demand).reshape((365,24,4))

        ### Real data
        self.energy_demand_real = copy.deepcopy(np.array(energy_demand)).reshape((365,24,3)) 
        self.heat_demand_real = copy.deepcopy(np.array(heat_demand)).reshape((365,24,4)) 
        self.gas_demand_real = copy.deepcopy(np.array(gas_demand)) 
        self.res_real = copy.deepcopy(np.array(res)) 
        self.wind_real = self.res_real[:,0].reshape((365,24))
        self.pv_real = self.res_real[:,1].reshape((365,24))

        ### Adding noise
        energy_demand = self.generate_normal_random_matrix(np.array(energy_demand), self.std)
        heat_demand = self.generate_normal_random_matrix(np.array(heat_demand), self.std)
        gas_demand = self.generate_normal_random_matrix(np.array(gas_demand), self.std)
        res = self.generate_normal_random_matrix(np.array(res), self.std)

        ### Noisy demand for observation
        self.energy_demand = np.array(energy_demand).reshape((365,24,3)) 
        self.gas_demand = np.array(gas_demand) 
        self.heat_demand = np.array(heat_demand).reshape((365,24,4)) 
        ### Noisy Renewable energy output
        self.wind = res[:,0].reshape((365,24)) 
        self.pv = res[:,1].reshape((365,24)) 

        ### Price
        self.electricity_price = np.array(elec_price)
        # self.gas_price = np.array(gas_price)
        self.gas_price = 0.20
        self.carbon_price = carbon_price
        
        ## Initialize states of each operators 
        initialized_episode_step = 0
        initialized_env_step = 0
        initial_states = {}
        for i,j in zip(self.operator_name, self.obs_num):
            initial_states[i] = np.zeros((j))

        ###  initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[0]][0] = 0
        initial_states[self.operator_name[0]][1] = self.electricity_price[initialized_env_step]               
        initial_states[self.operator_name[0]][2] = self.electricity_price[initialized_env_step] / 2         #购价为售价的2倍
        initial_states[self.operator_name[0]][3] = self.pv[initialized_episode_step, initialized_env_step]
        initial_states[self.operator_name[0]][4] = self.wind[initialized_episode_step, initialized_env_step]
        initial_states[self.operator_name[0]][5] = self.energy_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[0]][6] = self.cer_grid[initialized_episode_step, initialized_env_step]

        ### GSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[1]][0] = 0
        initial_states[self.operator_name[1]][1] = self.gas_demand[initialized_env_step].sum()


        ### HSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[2]][0] = 0
        initial_states[self.operator_name[2]][1] = self.heat_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[2]][2] = 0

        self.states = initial_states


    def step(self, epoch, action, trade, env_step, max_env_step):             
        action = action
        
        if self.test:
            iteration = epoch
        else:
            if self.oneday_flag:
                iteration = 0
            else:
                iteration = epoch % 334                                            # 前11个月为训练集，最后一个月为测试集

        ## output of devices
        output_dg = action[0] * self.dg_power_max
        output_gt = action[1] * self.gt_power_max
        output_es = action[2] * self.es_power_max

        output_gs = action[3] * self.gs_power_max

        output_ehp = action[4] * self.ehp_power_max
        output_gb = action[5] * self.gb_power_max
        output_ts = action[6] * self.ts_power_max

        if env_step >= max_env_step - 1:                                    
            done = True
        else:
            done = False

        if done:
            total_soc = self.states[self.operator_name[0]][0] + self.states[self.operator_name[1]][0] + self.states[self.operator_name[2]][0]

        ### PSO state transition
        if self.config[0] == 0:
            self.states[self.operator_name[0]][0] = 0
        else:
            if output_es <= 0:                         
                self.states[self.operator_name[0]][0] -= output_es / self.es_capacity * self.es_efficiency     
            else:
                self.states[self.operator_name[0]][0] -= output_es / self.es_capacity / self.es_efficiency
            
        if not done:
            self.states[self.operator_name[0]][1] = self.electricity_price[env_step+1] 
            self.states[self.operator_name[0]][2] = self.electricity_price[env_step+1] / 2
            self.states[self.operator_name[0]][3] = self.pv[iteration][env_step+1]
            self.states[self.operator_name[0]][4] = self.wind[iteration][env_step+1]
            self.states[self.operator_name[0]][5] = self.energy_demand[iteration][env_step+1].sum()
            self.states[self.operator_name[0]][6] = self.cer_grid[iteration][env_step+1]
        else:
            self.states[self.operator_name[0]][1] = 0 
            self.states[self.operator_name[0]][2] = 0 
            self.states[self.operator_name[0]][3] = 0 
            self.states[self.operator_name[0]][4] = 0 
            self.states[self.operator_name[0]][5] = 0 
            self.states[self.operator_name[0]][6] = 0 
        
        ### GSO state transition
        if self.config[1] == 0:
            self.states[self.operator_name[1]][0] = 0
        else:
            if output_gs <= 0:                             
                self.states[self.operator_name[1]][0] -= output_gs / self.gs_capacity * self.gs_efficiency    
            else:
                self.states[self.operator_name[1]][0] -= output_gs / self.gs_capacity / self.gs_efficiency
            
        if not done:
            self.states[self.operator_name[1]][1] = self.gas_demand[env_step+1].sum()
        else:
            self.states[self.operator_name[1]][1] = 0

        ### HSO state transition
        if self.config[2] == 0:
            self.states[self.operator_name[2]][0] = 0
        else:
            if output_ts <= 0:                              
                self.states[self.operator_name[2]][0] -= output_ts / self.ts_capacity * self.ts_efficiency     
            else:
                self.states[self.operator_name[2]][0] -= output_ts / self.ts_capacity / self.ts_efficiency
            
        if not done:
            self.states[self.operator_name[2]][1] = self.heat_demand[iteration][env_step+1].sum()
        else:
            self.states[self.operator_name[2]][1] = 0
        self.states[self.operator_name[2]][2] = env_step + 1

        ## Reward calculation
        energy_delta = trade[0]

        lambda_gas = self.gas_price

        if energy_delta >= 0:
            lambda_electricity = self.electricity_price[env_step]
        else:
            lambda_electricity = self.electricity_price[env_step] / 2

        ### PSO reward & cost
        
        reward_pso = - (lambda_electricity * energy_delta) - (self.cost_dg * output_dg) - (self.cost_gt * output_gt) 
        
        if energy_delta >= 0:
            reward_pso_carbon = self.carbon_price * (- (self.cer_grid[iteration][env_step] * energy_delta) - (self.cer_dg * output_dg) - (self.cer_gt * output_gt))
        else:
            reward_pso_carbon = self.carbon_price * (- (self.cer_dg * output_dg) - (self.cer_gt * output_gt))

        operation_cost_power = - (lambda_electricity * energy_delta) - (self.cost_dg * output_dg) - (self.cost_gt * output_gt)
        if energy_delta >= 0:
            carbon_cost_power = self.carbon_price * (- (self.cer_grid[iteration][env_step] * energy_delta) - (self.cer_dg * output_dg) - (self.cer_gt * output_gt))
        else:
            carbon_cost_power = self.carbon_price * (- (self.cer_dg * output_dg) - (self.cer_gt * output_gt))
        
        ### GSO reward & cost
        gas_delta = trade[1]
        
        reward_gso = - (lambda_gas * gas_delta)  
        
        operation_cost_gas = - (lambda_gas * gas_delta)

        ### HSO reward & cost
        reward_hso = - (self.cost_gb * output_gb) - (self.cost_ehp  * output_ehp)

        reward_hso_carbon = - (self.carbon_price * self.cer_gb * output_gb) 

        operation_cost_heat = - (self.cost_gb * output_gb) - (self.cost_ehp  * output_ehp)
        carbon_cost_heat = - (self.carbon_price * self.cer_gb * output_gb)

        # summation
        if self.alpha == 1.0:
            reward = (reward_pso_carbon + reward_hso_carbon) * 10
        else:
            reward = (((reward_pso + reward_gso + reward_hso) * (1-self.alpha)) + (self.alpha * (reward_pso_carbon + reward_hso_carbon))) * self.reward_scale
        operation_cost = operation_cost_power + operation_cost_gas + operation_cost_heat
        carbon_cost = carbon_cost_power + carbon_cost_heat

        state = self.flatten_states(self.states)                         

        return state, copy.deepcopy(state), reward, [operation_cost, carbon_cost], done                       

    def reset(self):
        ## Initialize states of each operators 
        initialized_episode_step = 0
        initialized_env_step = 0
        initial_states = {}
        for i,j in zip(self.operator_name, self.obs_num):
            initial_states[i] = np.zeros((j))

        ### PSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[0]][0] = 0
        initial_states[self.operator_name[0]][1] = self.electricity_price[initialized_env_step]             
        initial_states[self.operator_name[0]][2] = self.electricity_price[initialized_env_step] / 2           
        initial_states[self.operator_name[0]][3] = self.pv[initialized_episode_step, initialized_env_step]
        initial_states[self.operator_name[0]][4] = self.wind[initialized_episode_step, initialized_env_step]
        initial_states[self.operator_name[0]][5] = self.energy_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[0]][6] = self.cer_grid[initialized_episode_step, initialized_env_step]

        ### GSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[1]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[1]][0] = 0
        initial_states[self.operator_name[1]][1] = self.gas_demand[initialized_env_step].sum()

        ### HSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[2]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[2]][0] = 0
        initial_states[self.operator_name[2]][1] = self.heat_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[2]][2] = 0
        
        self.states = initial_states

        state = self.flatten_states(self.states)

        done = False

        return state, copy.deepcopy(state), done
    
    
    def reset_SOC(self, epoch):
        
        if self.test:
            iteration = epoch
        else:
            if self.oneday_flag:
                iteration = 0
            else:
                iteration = epoch % 334

        initialized_env_step = 0
        
        ### PSO reset SOC and transit to next state
        np.random.seed(self.seed)
        # self.states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        self.states[self.operator_name[0]][0] = 0
        self.states[self.operator_name[0]][1] = self.electricity_price[initialized_env_step] 
        self.states[self.operator_name[0]][2] = self.electricity_price[initialized_env_step] / 2
        self.states[self.operator_name[0]][3] = self.pv[iteration][initialized_env_step]
        self.states[self.operator_name[0]][4] = self.wind[iteration][initialized_env_step]
        self.states[self.operator_name[0]][5] = self.energy_demand[iteration][initialized_env_step].sum()
        self.states[self.operator_name[0]][6] = self.cer_grid[iteration][initialized_env_step]

        ### GSO reset SOC and transit to next state
        np.random.seed(self.seed)
        # self.states[self.operator_name[1]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        self.states[self.operator_name[1]][0] = 0
        self.states[self.operator_name[1]][1] = self.gas_demand[initialized_env_step].sum()

        ### HSO reset SOC and transit to next state
        np.random.seed(self.seed)
        # self.states[self.operator_name[2]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        self.states[self.operator_name[2]][0] = 0
        self.states[self.operator_name[2]][1] = self.heat_demand[iteration][initialized_env_step].sum()
        self.states[self.operator_name[2]][2] = 0

        state = self.flatten_states(self.states)
        done = False

        return state, copy.deepcopy(state), done

    def flatten_states(self, ori_states):
        for count, name in zip(range(len(self.operator_name)), self.operator_name):
            state = ori_states[name]
            if count == 0:
                flatten_states = state
            else:
                flatten_states = np.concatenate((flatten_states, state), axis = 0)
        return flatten_states

    def get_obs_dim(self):
        return int(self.obs_dim)

    def get_action_dim(self):
        return int(self.action_dim)
    
    def get_device_config(self):
        device_config = {}
        device_config['pso'] = np.array([self.dg_power_max, self.gt_power_max, self.es_power_max, self.es_capacity, self.es_efficiency, self.grid_max])
        device_config['gso'] = np.array([self.gs_power_max, self.gs_capacity, self.gs_efficiency, self.gas_max])
        device_config['hso'] = np.array([self.ehp_power_max, self.gb_power_max, self.ts_power_max, self.ts_capacity, self.ts_efficiency])

        return device_config
    
    def get_load(self, epoch, env_step):                
        
        if self.test:
            iteration = epoch
        else:
            if self.oneday_flag:
                iteration = 0
            else:
                iteration = epoch % 334
            
        load = {}
        load['pso'] = self.energy_demand_real[iteration][env_step]
        load['gso'] = self.gas_demand_real[env_step]
        load['hso'] = self.heat_demand_real[iteration][env_step]

        return load

    def get_res_output(self, epoch, env_step):          
        
        if self.test:
            iteration = epoch
        else:
            if self.oneday_flag:
                iteration = 0
            else:
                iteration = epoch % 334
            
        res_output = {}
        res_output['wind'] = self.wind_real[iteration][env_step]
        res_output['pv'] = self.pv_real[iteration][env_step]

        return res_output
    
    def generate_normal_random_matrix(self, input_matrix, std):               
        length = input_matrix.shape[0]
        width = input_matrix.shape[1]
        output_matrix = np.zeros((length, width))
        for i in range(length):
            for j in range(width):
                output_matrix[i,j] = np.random.normal(input_matrix[i,j], std*input_matrix[i,j])
        
        return output_matrix

    def cost_calculation(self, epoch, env_step, state, action):       
        load = self.get_load(epoch, env_step)
        res_output = self.get_res_output(epoch, env_step)

        ## demand
        energy_demand = load['pso'].sum()
        gas_demand = load['gso'].sum() 
        heat_demand = load['hso'].sum()
        ## RES generation
        pv_output = res_output['pv']
        wt_output = res_output['wind']
        ## SoC
        SOC_es = state[0]
        SOC_gs = state[7]    
        SOC_ts = state[9]

        safe_action = copy.deepcopy(action)
        device_config = self.get_device_config()

        # current action    
        action_dg = action[0] * device_config['pso'][0]                
        action_gt = action[1] * device_config['pso'][1]
        action_es = action[2] * device_config['pso'][2]                
        
        action_gs = action[3] * device_config['gso'][0]                
        
        action_ehp = action[4] * device_config['hso'][0]
        action_gb = action[5] * device_config['hso'][1]
        action_ts = action[6] * device_config['hso'][2]              

        ## overcharging or overdischarging
        ### energy storage
        cost3 = 0
        maxp_storage_es = device_config['pso'][2]
        capacity_storage_es = device_config['pso'][3]
        efficiency_es = device_config['pso'][4]        
        ubp_storage_es = min(capacity_storage_es * SOC_es * efficiency_es, maxp_storage_es)      
        lbp_storage_es = max((SOC_es - 1) * capacity_storage_es /efficiency_es, -maxp_storage_es)   
        if action_es <= ubp_storage_es and action_es >= lbp_storage_es:
            action_es = action_es
        elif action_es > ubp_storage_es:
            cost3 = action_es - ubp_storage_es
            action_es = ubp_storage_es
        elif action_es < lbp_storage_es:
            cost3 = lbp_storage_es - action_es
            action_es = lbp_storage_es
        if self.config[0] == 0:
            cost3 = 0
        else:
            safe_action[2] = action_es/ device_config['pso'][2]

        ### gas storage
        cost4 = 0
        maxp_storage_gs = device_config['gso'][0]
        capacity_storage_gs = device_config['gso'][1]
        efficiency_gs = device_config['gso'][2]        
        ubp_storage_gs = min(capacity_storage_gs * SOC_gs * efficiency_gs, maxp_storage_gs)
        lbp_storage_gs = max((SOC_gs - 1) * capacity_storage_gs /efficiency_gs, -maxp_storage_gs)
        if action_gs <= ubp_storage_gs and action_gs >= lbp_storage_gs:
            action_gs = action_gs
        elif action_gs > ubp_storage_gs:
            cost4 = action_gs - ubp_storage_gs
            action_gs = ubp_storage_gs
        elif action_gs < lbp_storage_gs:
            cost4 = lbp_storage_gs - action_gs
            action_gs = lbp_storage_gs
        if self.config[1] == 0:
            cost4 = 0
        else:
            safe_action[3] = action_gs / device_config['gso'][0]
                
        ### thermal storage
        cost5 = 0
        maxp_storage_ts = device_config['hso'][2]
        capacity_storage_ts = device_config['hso'][3]
        efficiency_ts = device_config['hso'][4]        
        ubp_storage_ts = min(capacity_storage_ts * SOC_ts * efficiency_ts, maxp_storage_ts)
        lbp_storage_ts = max((SOC_ts - 1) * capacity_storage_ts /efficiency_ts, -maxp_storage_ts)
        if action_ts <= ubp_storage_ts and action_ts >= lbp_storage_ts:
            action_ts = action_ts
        elif action_ts > ubp_storage_ts:
            cost5 = action_ts - ubp_storage_ts
            action_ts = ubp_storage_ts
        elif action_ts < lbp_storage_ts:
            cost5 = lbp_storage_ts - action_ts
            action_ts = lbp_storage_ts
        if self.config[2] == 0:
            cost5 = 0
        else:
            safe_action[6] = action_ts / device_config['hso'][2]
        
        # energy trading calculation
        action_upg = (energy_demand + action_ehp/3) - wt_output - pv_output - action_dg - action_gt - action_es
        action_ugg = (gas_demand + action_gt/2 + action_gb/5) - action_gs                

        # action correction & cost calculation
        ## electricity demand unbalance: i) import limit ii) export limit
        cost1 = 0
        res_curtailment = 0
        if action_upg > 0:
            cv_upg = action_upg - device_config['pso'][5]
            if cv_upg > 0:
                cost1 = cv_upg
                action_upg = device_config['pso'][5]
        if action_upg < 0:
            cv_upg =  abs(action_upg) - device_config['pso'][5]
            if cv_upg > 0 and (cv_upg-pv_output-wt_output) > 0:
                cost1 = cv_upg-pv_output-wt_output
                action_upg = - device_config['pso'][5]
                res_curtailment = pv_output + wt_output
            elif cv_upg > 0 and (cv_upg-pv_output-wt_output) < 0:
                action_upg = - device_config['pso'][5]
                res_curtailment = cv_upg

        ## gas demand unbalance: i) import limit ii) export limit
        cost6 = 0
        if action_ugg > 0:
            cv_ugg = action_ugg - device_config['gso'][3]
            if cv_ugg > 0:
                cost6 = cv_ugg
                action_ugg = device_config['gso'][3]

        ## heat unbalance
        cost2 = 0
        heat_unbalance = heat_demand - action_ehp - action_gb - action_ts
        cost2 = abs(heat_unbalance)
        
        safe_trade = np.array([action_upg, action_ugg, res_curtailment])
        cost =[cost1, cost2, cost3, cost4, cost5, cost6]


        return safe_action, safe_trade, cost             


