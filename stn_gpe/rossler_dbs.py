import numpy as np
import matplotlib.pyplot as plt


# Euler's Methods for integration
def euler_method(dfdt, f, dt):
    f = f + dfdt*dt
    return f

# Rossler Network Equations
def derivative(state, X, wk, P = 0, psi = 0, k = 0.45):
    x, y, z = state
    a = 0.2
    b = 0.2
    c = 5.7
    d = 8 #0.5
    omega = 0.8 
    I_ext = 0 #0.1
    dxdt = -omega*y - z + k*0.05*(d-X) +(0.5-k)*(0.5*X-x) + np.cos(psi)*P
    dydt = omega*x + a*y + np.sin(psi)*P
    dzdt = b + z*(x-c) + I_ext
    return dxdt, dydt, dzdt


# Biphasic open loop pulses
def open_loop_control(t, dbs_frequency, P_amp, pos_width, neg_width, dt = 0.01):

    T = 1/dbs_frequency # Period of the pulse train
    pos_gap = (T - pos_width - neg_width)/2
    neg_gap = (T - pos_width - neg_width)/2
    P = 0
    if t%((1/dt)*(pos_width+pos_gap+neg_width+neg_gap)) < (1/dt)*pos_width:
        P = P_amp
    elif t%((1/dt)*(pos_width+pos_gap+neg_width+neg_gap)) < (1/dt)*(pos_width + pos_gap):
        P = 0
    elif t%((1/dt)*(pos_width+pos_gap+neg_width+neg_gap)) < (1/dt)*(pos_width + pos_gap + neg_width):
        P = -P_amp
    else:
        P = 0
    
    return P


# Phase Estimation for closed loop pulses
def phase_est(u, u_dot, d, dt, X, omega_0 = 1, damping_coef = 0.3, mu = 500):
    alpha = damping_coef*omega_0

    u_dot = u_dot + dt*(X-alpha*u_dot-omega_0*omega_0*u)
    u = u + dt*u_dot
    d = d + dt*(u_dot-d)/mu

    x_cap = alpha*u_dot
    y_cap = alpha*omega_0*mu*d

    amp = np.sqrt(x_cap**2 + y_cap**2)

    theta = np.arctan2(y_cap, x_cap) # value between -pi and pi
    theta = theta if theta >= 0 else theta + 2*np.pi # normalizing theta between 0 and 2*pi

    return u, u_dot, d, amp, theta, x_cap, y_cap

# abar for closed loop pulses
def abar(theta_peaks, theta_01, theta, theta_tol, amp):
    a_bar_value = 0
    j = 0
    for i in range(theta_peaks[-2], theta_peaks[-1], 1):
        # if (abs(theta[-1] - theta_01) < theta_tol or abs(theta[-1] - theta_01 - np.pi) < theta_tol) == False:
        #     a_bar_value = a_bar_value + amp[i]
        #     j = j + 1
        a_bar_value = a_bar_value + amp[i]
        j = j + 1

    average_a_bar = a_bar_value/j
    return average_a_bar


def ATSP(a_stop, a_bar, theta_0, epsilon_fb, k1 = 0.001, k2 = 500, k3 = 0.001, k4 = 5, dt = 0.01):
    S = k1*a_bar*(1+np.tanh(k2*(a_bar - a_stop)))
    T = -k3*a_bar/np.cosh(k4*epsilon_fb)
    theta_0 = theta_0 + 0.1*S
    epsilon_fb = epsilon_fb + T
    return theta_0, epsilon_fb

def control_input(amp, theta, theta_0, epsilon_fb, flag = False, A0 = 3, delta = 0.2, delta_cap = 0.4, theta_tol = 0.04*np.pi):
    control = max(epsilon_fb*amp, -A0)

    if abs(theta - theta_0 - np.pi) < theta_tol:
        control = -control
    
    return control

# area finder
def area_finder(current_area, dt, amp_with_sign):
    area = current_area + amp_with_sign*dt
    return area

# Run closed loop DBS
def run_closed_loop_DBS(num_oscillators, t_eval, 
                        wk, DBS_time, 
                        theta_tol = 0.15*np.pi, 
                        dt = 0.01, k1_val = 0.001, 
                        k3_val = 0.001, k_val = 0.45):
    print(f"Running closed loop DBS for k = {k_val}")
    initial_conditions = np.zeros((num_oscillators, 3))

    # Modify the 'x' values to be random from a normal distribution
    initial_conditions[:, 0] = np.random.normal(0, 0.001, num_oscillators)
    # y and z remain zeros (no change needed)
    global_states = np.mean(initial_conditions, axis=0)
    global_X = np.array([global_states[0]])
    global_Y = np.array([global_states[1]])
    global_Z = np.array([global_states[2]])
    u = 0
    u_dot = 0
    d = 0
    a_stop = 0
    theta_0 = 0
    theta_0_arr = [0]
    epsilon_fb = 0
    epsilon_fb_arr = [0]
    P_array = [0]
    amp_arr = []
    theta_arr = [0]
    theta_peak = []
    size_theta_peak = 0
    current_delta = 0
    delta = 0.4/dt
    current_Delta = 0
    Delta = 0.2/dt
    j_critical = -100
    P_pos_avg = 0
    P_neg_avg = 0
    delta_neg = 0
    j_neg = 0
    Area_total = 0
    number_of_cycles = 0
    x_arr_4 = [] # 2D matrix containing x values of  first 4 oscillators
    y_arr_4 = [] # 2D matrix containing y values of  first 4 oscillators
    z_arr_4 = [] # 2D matrix containing z values of  first 4 oscillators
    x_4 = [] # x values of  first 4 oscillators at a particular time
    y_4 = [] # y values of  first 4 oscillators at a particular time
    z_4 = [] # z values of  first 4 oscillators at a particular time
    for i in range(4):
        x_4.append(initial_conditions[i][0])
        y_4.append(initial_conditions[i][1])
        z_4.append(initial_conditions[i][2])
    x_arr_4.append(x_4)
    y_arr_4.append(y_4)
    z_arr_4.append(z_4)

    for j in range(np.size(t_eval)-1):
        if j%10000 == 0:
            print("time = ", j*dt)
        if j <= int(DBS_time/dt):
            u, u_dot, d, amp, theta, x_cap, y_cap = phase_est(u, u_dot, d, dt, global_X[-1])
            P = 0
            P_array.append(P)
            amp_arr.append(amp)
            theta_arr.append(theta)
            if j == int(DBS_time/dt):
                a_stop = 0.2*np.mean(amp_arr)
        else:
            P = 0
            amp = 0
            u, u_dot, d, amp, theta, x_cap, y_cap = phase_est(u, u_dot, d, dt, global_X[-1])
            flag = False
            if abs(theta - theta_0) < theta_tol:
                flag = True
                if j_critical == -100:
                    j_critical = j
            
            if flag == False:
                j_critical = -100

            if flag == True:
                P = control_input(amp, theta, theta_0, epsilon_fb, flag)

            if flag == True:
                if abs(theta - theta_0) < theta_tol:
                    if (j - j_critical)%(delta + Delta) < delta:
                        P = P
                    else:
                        P = 0
            
            if theta_arr[-1] < np.pi and theta >=np.pi:
                delta_pos = 0
                Area_pos = 0
                for s in range(theta_peak[-1], j, 1):
                    Area_pos = P_array[s]*dt + Area_pos
                    if P_array[s] != 0:
                        delta_pos = delta_pos + 1
                    
                P_pos_avg = Area_pos/(delta_pos*dt + 10e-7)
                P_neg_avg = -P_pos_avg/5
                delta_neg = abs(Area_total/(P_neg_avg*dt + 10e-7))
                j_neg = j + delta_neg
                
            if theta_arr[-1] > np.pi:
                if j <= j_neg:
                    P = P_neg_avg

            Area_total = Area_total + P*dt
            P_array.append(P)
            amp_arr.append(amp)
            theta_arr.append(theta)


        if np.size(theta_arr) > 3 and theta_arr[-1] < theta_arr[-2] and theta_arr[-2] > theta_arr[-3]:
            number_of_cycles = number_of_cycles + 1
            theta_peak.append(j-1)
            current_delta = 0
            size_theta_peak = size_theta_peak + 1 
            if j >= int(DBS_time/dt):
                a_bar = abar(theta_peak, theta_0, theta_arr, theta_tol, amp_arr)
                theta_0, epsilon_fb = ATSP(a_stop, a_bar, theta_0, epsilon_fb, k1 = k1_val, k3 = k3_val)
                theta_0_arr.append(theta_0)
                epsilon_fb_arr.append(epsilon_fb)
            else:
                theta_0_arr.append(0)
                epsilon_fb_arr.append(0)
            
            
        x_4 = []
        y_4 = []
        z_4 = []
        for i in range(num_oscillators):
            state = initial_conditions[i]
            dxdt, dydt, dzdt = derivative(state, global_states[0], wk[i], P, psi=np.pi/4, k = k_val)
            x = euler_method(dxdt, state[0], dt)
            y = euler_method(dydt, state[1], dt)
            z = euler_method(dzdt, state[2], dt)
            if i < 4:
                x_4.append(x)
                y_4.append(y)
                z_4.append(z)
            initial_conditions[i][0] = float(x)
            initial_conditions[i][1] = float(y)
            initial_conditions[i][2] = float(z)

        x_arr_4.append(x_4)
        y_arr_4.append(y_4)
        z_arr_4.append(z_4)
        global_states = np.mean(initial_conditions, axis=0)
        global_X = np.append(global_X, global_states[0])
        global_Y = np.append(global_Y, global_states[1])
        global_Z = np.append(global_Z, global_states[2])


    return global_X, P_array, theta_arr, epsilon_fb_arr, theta_0_arr, global_Y, global_Z, x_arr_4, y_arr_4, z_arr_4


pos_width = [0.4]
frequency = [0.2]
amplitude = [0.2, 0.5, 0.8, 1, 2]

def run_open_loop_DBS(t_eval, wk,  k_val = 0.45, freq = 16, P_a = 0.8, dt = 0.01, p_w = 0.4, n_w = 0.4, DBS_time = 1000, num_oscillators = 100):
    print(f"Running Open Loop DBS for k = {k_val}")
    initial_conditions = np.zeros((num_oscillators, 3))
    initial_conditions[:, 0] = np.random.normal(0, 0.001, num_oscillators)
    global_states = np.mean(initial_conditions, axis=0)
    global_X = np.array([global_states[0]])
    global_Y = np.array([global_states[1]])
    global_Z = np.array([global_states[2]])
    P_array = [0]
    pos_width = p_w
    neg_width = n_w
    freq*=0.0125 
    P_amp = P_a
    x_arr_10 = [] # monitor to store x values of  first 10 oscillators for the entire simulation
    x_10 = [] # x values of  first 10 oscillators at a particular time
    for i in range(10):
        x_10.append(initial_conditions[i][0])

    x_arr_10.append(x_10)

    for j in range(np.size(t_eval)-1):
        if j%10000 == 0:
            print("time = ", j*dt)
        if j <= int(DBS_time/dt):
            P = 0
            P_array.append(P)
        else:
            P = open_loop_control(j, freq, P_amp, pos_width, neg_width, dt)
            P_array.append(P)  
        x_10 = []
        for i in range(num_oscillators):
            state = initial_conditions[i]
            dxdt, dydt, dzdt = derivative(state, global_states[0], wk[i], P, psi=np.pi/4, k = k_val)
            x = euler_method(dxdt, state[0], dt)
            y = euler_method(dydt, state[1], dt)
            z = euler_method(dzdt, state[2], dt)
            if i < 10:
                x_10.append(x)
            initial_conditions[i][0] = float(x)
            initial_conditions[i][1] = float(y)
            initial_conditions[i][2] = float(z)

        x_arr_10.append(x_10)
        global_states = np.mean(initial_conditions, axis=0)
        global_X = np.append(global_X, global_states[0])
        global_Y = np.append(global_Y, global_states[1])
        global_Z = np.append(global_Z, global_states[2])

    return global_X, P_array, global_Y, global_Z, x_arr_10


def run_std_DBS(t_eval, wk,  k_val = 0.45, freq = 130, P_a = 2, dt = 0.01, p_w = 0.75, n_w = 0.75, DBS_time = 1000, num_oscillators = 100):
# def run_std_DBS(num_oscillators, t_eval, wk, DBS_time, p_w, n_w, freq, P_a, k_val = 0.45, dt = 0.01):
    print(f"Running standard DBS for k = {k_val}")
    initial_conditions = np.zeros((num_oscillators, 3))
    initial_conditions[:, 0] = np.random.normal(0, 0.001, num_oscillators)
    global_states = np.mean(initial_conditions, axis=0)
    global_X = np.array([global_states[0]])
    global_Y = np.array([global_states[1]])
    global_Z = np.array([global_states[2]])
    P_array = [0]
    pos_width = p_w
    neg_width = n_w
    freq *= 0.005
    P_amp = P_a
    x_arr_10 = [] # monitor to store x values of  first 10 oscillators for the entire simulation
    x_10 = [] # x values of  first 10 oscillators at a particular time
    for i in range(10):
        x_10.append(initial_conditions[i][0])
    x_arr_10.append(x_10)


    for j in range(np.size(t_eval)-1):
        if j%10000 == 0:
            print("time = ", j*dt)
        if j <= int(DBS_time/dt):
            P = 0
            P_array.append(P)
        else:
            P = open_loop_control(j, freq, P_amp, pos_width, neg_width, dt)
            P_array.append(P)

        x_10 = []
        for i in range(num_oscillators):
            state = initial_conditions[i]
            dxdt, dydt, dzdt = derivative(state, global_states[0], wk[i], P, psi=np.pi/4, k = k_val)
            x = euler_method(dxdt, state[0], dt)
            y = euler_method(dydt, state[1], dt)
            z = euler_method(dzdt, state[2], dt)
            if i < 10:
                x_10.append(x)
            initial_conditions[i][0] = float(x)
            initial_conditions[i][1] = float(y)
            initial_conditions[i][2] = float(z)

        x_arr_10.append(x_10)
        global_states = np.mean(initial_conditions, axis=0)
        global_X = np.append(global_X, global_states[0])
        global_Y = np.append(global_Y, global_states[1])
        global_Z = np.append(global_Z, global_states[2])

    return global_X, P_array, global_Y, global_Z, x_arr_10
