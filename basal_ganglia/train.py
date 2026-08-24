import sys, os

# Set project root to ROSSELER_CLEANED_FILES (parent of basal_ganglia/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)


# Path to STN-GPe parameters directory
STN_GPE_PARAMS_FILE_PATH = os.path.join(project_root, "params", "stn_gpe_params", "params_k.yaml")


from .BGNetwork import BGNetwork
from stn_gpe import RosslerNetwork
import torch
import numpy as np
from tqdm.autonotebook import tqdm
import yaml

def update_epsilon(ep_old,TD_error,alpha_ep, eta_ep, baseline_val = 0.02):
    ep = ep_old + alpha_ep * (1- torch.exp((-TD_error**2)/eta_ep) - ep_old) + baseline_val
    return ep.item()

def run_STN_GPe_system(ep, osc_no,
                       m = 0.5, #0.6, 
                       c = 0.4, #0.3,
                       time_sec = 5.5, 
                       k_lim = None, 
                       stn_gpe_params_path = STN_GPE_PARAMS_FILE_PATH):
    
    with open(stn_gpe_params_path,'r') as file:
        data = yaml.safe_load(file)
    omega =np.ones((data['N'],1))*data['omega']   
    if k_lim is None: 
        k = m * ep + c
    else:
        k = k_lim

    if k >= 0.55:
        k = 0.55 # safety bounds to k to avoid overflow
    elif k<=0.25:
        k = 0.25 # safety bounds to k to avoid overflow

    RN = RosslerNetwork(N = data['N'], 
                    a = data['a'],
                    b = data['b'], 
                    c = data['c'], 
                    d = data['d'], 
                    k = k,
                    omega= omega, 
                    Iext=data['Iext'])
    sol = RN.run(time_sec=time_sec)
    sampling_rate = 50000
    t_low = int(sampling_rate * (time_sec-2))
    t_high = int(sampling_rate * time_sec) 
    random_start = np.random.choice(np.arange(50))
    x_vals = np.array(sol.y)[0:data['N'],t_low:t_high] 
    osc_output = np.mean(x_vals[osc_no].reshape(-1,200,500),2) #(osc, time)
    normalized_x_vals = (osc_output - np.min(osc_output))/(np.max(osc_output)-np.min(osc_output))
    x_std = np.std(normalized_x_vals,0)

    x_data_ = normalized_x_vals[:,random_start:random_start+50] + 0.5 #converting mean to be 1

    x_rand = torch.randn((normalized_x_vals.shape[0],normalized_x_vals.shape[1]), requires_grad= False) * ep + 1 
    x_data =  x_rand[:,random_start:random_start+50]
    return x_data, np.mean(x_std)


def run_STN_GPe_closedloop(num_arms, data = None):
    data_path = os.path.join(project_root, 'basal_ganglia', 'rossler_dbs_stn_data', 'closed_loop')
    x_osc = np.loadtxt(os.path.join(data_path, 'x_osc_closedloop.txt'), delimiter=',')
    x_osc = x_osc[299999:399999, 0:num_arms]  # extracting last 10 sec data # (time, osc)
    osc_output = x_osc.reshape(-1, 100, x_osc.shape[1]).mean(axis=1)
    normalized_x_vals = (osc_output - np.min(osc_output))/(np.max(osc_output)-np.min(osc_output))
    normalized_x_vals = normalized_x_vals.T #(osc, time)
    x_std = np.std(normalized_x_vals, axis=0)
    random_start = np.random.choice(np.arange(800))
    x_data_ = normalized_x_vals[:,random_start:random_start+50] + 0.5 #converting mean to be 1
    x_rand = torch.randn((normalized_x_vals.shape[0],normalized_x_vals.shape[1]), requires_grad= False) * np.mean(x_std) + 1 
    x_data =  x_rand[:,random_start:random_start+50]
    return x_data, np.mean(x_std)

def run_STN_GPe_standard_dbs(num_arms, data = None):
    data_path = os.path.join(project_root, 'basal_ganglia', 'rossler_dbs_stn_data', 'standard_dbs')
    x_osc = np.loadtxt(os.path.join(data_path, 'x_osc_amp_2.txt'), delimiter=',')
    x_osc = x_osc[299999:399999, 0:num_arms]  # extracting last 10 sec data # (time, osc)

    osc_output = x_osc.reshape(-1, 100, x_osc.shape[1]).mean(axis=1)
    normalized_x_vals = (osc_output - np.min(osc_output))/(np.max(osc_output)-np.min(osc_output))
    normalized_x_vals = normalized_x_vals.T #(osc, time)
    x_std = np.std(normalized_x_vals, axis=0)
    random_start = np.random.choice(np.arange(800))
    x_data_ = normalized_x_vals[:,random_start:random_start+50] + 0.5 #converting mean to be 1
    x_rand = torch.randn((normalized_x_vals.shape[0],normalized_x_vals.shape[1]), requires_grad= False) * np.mean(x_std) + 1 
    x_data =  x_rand[:,random_start:random_start+50]
    return x_data, np.mean(x_std)

def run_STN_GPe_openloop_low(num_arms, data = None):
    data_path = os.path.join(project_root, 'basal_ganglia', 'rossler_dbs_stn_data', 'open_loop_low_amp')
    x_osc = np.loadtxt(os.path.join(data_path, 'x_osc_amp_0.2.txt'), delimiter=',')
    x_osc = x_osc[299999:399999, 0:num_arms]  # extracting last 10 sec data
    osc_output = x_osc.reshape(-1, 100, x_osc.shape[1]).mean(axis=1)
    normalized_x_vals = (osc_output - np.min(osc_output))/(np.max(osc_output)-np.min(osc_output))
    normalized_x_vals = normalized_x_vals.T #(osc, time)
    x_std = np.std(normalized_x_vals, axis=0) 
    random_start = np.random.choice(np.arange(800))
    x_data_ = normalized_x_vals[:,random_start:random_start+50] + 0.5 #converting mean to be 1
    x_rand = torch.randn((normalized_x_vals.shape[0],normalized_x_vals.shape[1]), requires_grad= False) * np.mean(x_std) + 1 
    x_data =  x_rand[:,random_start:random_start+50]
    return x_data, np.mean(x_std)

def run_STN_GPe_openloop_high(num_arms, data = None):
    data_path = os.path.join(project_root, 'basal_ganglia', 'rossler_dbs_stn_data', 'open_loop_high_amp')
    x_osc = np.loadtxt(os.path.join(data_path, 'x_osc_amp_0.8.txt'), delimiter=',')
    x_osc = x_osc[299999:399999, 3:3+num_arms]  # extracting last 10 sec data
    osc_output = x_osc.reshape(-1, 100, x_osc.shape[1]).mean(axis=1)
    normalized_x_vals = (osc_output - np.min(osc_output))/(np.max(osc_output)-np.min(osc_output))
    normalized_x_vals = normalized_x_vals.T #(osc, time)
    x_std = np.std(normalized_x_vals, axis=0)
    random_start = np.random.choice(np.arange(800))
    x_data_ = normalized_x_vals[:,random_start:random_start+50] + 0.5 #converting mean to be 1
    x_rand = torch.randn((normalized_x_vals.shape[0],normalized_x_vals.shape[1]), requires_grad= False) * np.mean(x_std) + 1 
    x_data =  x_rand[:,random_start:random_start+50]
    return x_data, np.mean(x_std)


def train(env, trails, epochs, bins, lr , 
          d1_amp = 1, d2_amp = 5, gpi_threshold = 3, max_gpi_iters = 250,
          STN_data = None, del_lim = None, del_med = None, train_IP = True,
          printing = False, gpi_mean = 1,
          ep_0 = 0.5,alpha_ep = 0.25,eta_ep = 0.1, ep_lim = None, 
          baseline_ep = 0.02, k_lim = None, dbs_cond = None):
    
    num_arms = env.num_arms
    picks_per_bin = int(trails//bins)
    arm_chosen_monitor = torch.zeros(epochs,trails)

    reward_monitor = torch.zeros(epochs,trails)
    avg_counts = {i: torch.zeros(epochs,bins,1) for i in np.arange(num_arms)}
    ip_monitor = {i: torch.zeros(epochs,trails,1) for i in np.arange(num_arms)}
    dp_monitor = {i: torch.zeros(epochs,trails,1) for i in np.arange(num_arms)}
    ep_monitor = torch.zeros(epochs, trails)

    for epoch in range(epochs):
        # print(f'**************************{epoch}*************************************')
        ep = ep_0
        if ep_lim is not None:
            ep = torch.clamp(torch.tensor(ep), max = ep_lim).item()
        env.reset()
        bg_network = BGNetwork(max_gpi_iters = max_gpi_iters, 
                               d1_amp = d1_amp, 
                               d2_amp = d2_amp, 
                               gpi_threshold = gpi_threshold,
                               seed = epoch,
                               num_arms=env.num_arms)
        
        optimizer = torch.optim.Adam(params = bg_network.parameters(), lr = lr)
        
        for trail in range(trails):
            ep_monitor[epoch, trail] = ep
            bin_num = int(trail//picks_per_bin)
            random_osc_numbers = np.random.choice(np.arange(100), num_arms)
          
            if STN_data is None:
                stn_output = torch.randn((1,max_gpi_iters,num_arms), requires_grad= False) * ep + gpi_mean 
                # print(stn_output)

            elif STN_data == 'rossler' and dbs_cond is None:
                stn_output, _ = run_STN_GPe_system(ep = ep,osc_no = random_osc_numbers, k_lim = k_lim)
                stn_output = torch.tensor(stn_output.T,dtype=torch.float32).unsqueeze(0)

            elif STN_data == 'rossler' and dbs_cond is not None:
                print('running dbs cond ', dbs_cond)
                if dbs_cond == 'closed_loop':
                    stn_output, _ = run_STN_GPe_closedloop(num_arms = num_arms)
                if dbs_cond == 'open_loop_low':
                    stn_output, _ = run_STN_GPe_openloop_low(num_arms = num_arms)
                if dbs_cond == 'open_loop_high':
                    stn_output, _ = run_STN_GPe_openloop_high(num_arms = num_arms)
                if dbs_cond == 'standard_dbs':
                    stn_output, _ = run_STN_GPe_standard_dbs(num_arms = num_arms)
                stn_output = torch.tensor(stn_output.T,dtype=torch.float32).unsqueeze(0)
                
            
               
            gpi_out, gpi_iters, dp_output, ip_output = bg_network(stn_output)
            arm_chosen = torch.argmax(gpi_out)

            avg_counts[arm_chosen.item()][epoch,bin_num] = avg_counts[arm_chosen.item()][epoch,bin_num] + 1
            
            for arm in range(num_arms):
                ip_monitor[arm][epoch,trail] = ip_output[0,arm]
                dp_monitor[arm][epoch,trail] = dp_output[0,arm]

            reward = env.step(arm_chosen.item())
            TD_error =  reward - dp_output[:, arm_chosen] 
            if train_IP == False:
                for param in bg_network.D2_pathway.parameters():
                    param.requires_grad = False  
            
            if del_lim is not None:             
                TD_error = torch.clamp(TD_error, max=del_lim)
            
            if del_med is not None:
                TD_error = TD_error + del_med

            loss = TD_error**2    
            
            # setting gradients to zero
            optimizer.zero_grad()
            
            # Computing gradient
            loss.backward()
            
            # Updating weights
            optimizer.step()

            #network weights to clamped to only positive
            with torch.no_grad():
                for param in bg_network.parameters():
                    param.clamp_(min=0)  

            arm_chosen_monitor[epoch, trail] = arm_chosen.item()
            reward_monitor[epoch, trail] = reward

            ep = update_epsilon(ep_old=ep, TD_error=TD_error.detach(), alpha_ep=alpha_ep, eta_ep=eta_ep, baseline_val = baseline_ep)
            if ep_lim is not None:
                ep = torch.clamp(torch.tensor(ep), max = ep_lim).item()

            if printing:
                print(f'{trail}: dp: {dp_output}, arm_chosen:{arm_chosen}, TD error: {TD_error}, epsilon: {ep}')#dp_output, ip_output, gpi_out, gpi_iters,  arm_chosen, reward, TD_error)
    return reward_monitor, arm_chosen_monitor,avg_counts,ip_monitor, dp_monitor, ep_monitor


