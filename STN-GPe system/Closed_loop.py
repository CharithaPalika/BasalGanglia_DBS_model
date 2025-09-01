import numpy as np
import matplotlib.pyplot as plt
import yaml
from Rossler import euler_method, derivative

# --- Load Configuration from YAML file ---
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
ross_p = config['rossler_params']
est_p = config['phase_estimator_params']
atsp_p = config['atsp_feedback_params']
ctrl_p = config['control_input_params']
sim_p = config['closed_loop_simulation_params']


def phase_est(u, u_dot, d, dt, X, omega_0=est_p['omega_0'], damping_coef=est_p['damping_coef'], mu=est_p['mu']):
    """Estimates the instantaneous phase and amplitude of the input signal X."""
    alpha = damping_coef * omega_0
    
    u_dot = u_dot + dt * (X - alpha * u_dot - omega_0 * omega_0 * u)
    u = u + dt * u_dot
    d = d + dt * (u_dot - d) / mu

    x_cap = alpha * u_dot
    y_cap = alpha * omega_0 * mu * d

    amp = np.sqrt(x_cap**2 + y_cap**2)
    theta = np.arctan2(y_cap, x_cap)

    # Normalize theta to be between 0 and 2*pi
    if theta < 0:
        theta += 2 * np.pi
    
    return u, u_dot, d, amp, theta, x_cap, y_cap



def abar(theta_peaks, amp_arr):
    """Calculates the average amplitude over the last complete oscillation cycle."""
    if len(theta_peaks) < 2:
        return 0  # Can't calculate a cycle with less than two peaks

    start_index = theta_peaks[-2]
    end_index = theta_peaks[-1]
    
    num_points = end_index - start_index
    if num_points <= 0:
        return 0

    # Sum the amplitude values between the last two peaks
    total_amp = sum(amp_arr[start_index:end_index])
    
    return total_amp / num_points



def ATSP(a_stop, a_bar, theta_0, epsilon_fb, k1=atsp_p['k1'], k2=atsp_p['k2'], k3=atsp_p['k3'], k4=atsp_p['k4']):
    """Updates control parameters (theta_0, epsilon_fb) using the adaptive algorithm."""
    # S updates the target phase, T updates the feedback gain.
    S = k1 * a_bar * (1 + np.tanh(k2 * (a_bar - a_stop)))
    T = -k3 * a_bar / np.cosh(k4 * epsilon_fb)
    
    theta_0 += S
    epsilon_fb += T
    
    return theta_0, epsilon_fb



def control_input(amp, theta, theta_0, epsilon_fb, A0=ctrl_p['A0'], theta_tol_factor=ctrl_p['theta_tol_factor']):
    """Calculates the control input P based on the system's current state."""
    control = max(epsilon_fb * amp, -A0)
    theta_tol = theta_tol_factor * np.pi

    # If the system is in the anti-phase window, flip the control signal.
    if abs(theta - theta_0 - np.pi) < theta_tol:
        control = -control
    
    return control



def main(num_oscillators, t_eval):
    """The main function to run the entire simulation."""
    print("Starting main simulation...")
    print(f"Coupling strength k = {ross_p['k']}")

    # --- 1. Initialization ---
    dt = sim_p['dt']
    DBS_time = sim_p['DBS_time']
    
    # Set initial conditions for all oscillators
    initial_conditions = np.zeros((num_oscillators, 3))
    initial_conditions[:, 0] = np.random.normal(0, 0.001, num_oscillators)
    
    # Initialize arrays and variables for tracking simulation state
    global_states = np.mean(initial_conditions, axis=0)
    global_X = np.array([global_states[0]])
    global_Y = np.array([global_states[1]])
    global_Z = np.array([global_states[2]])
    
    # State variables for the phase estimator
    u, u_dot, d = 0, 0, 0
    
    # Adaptive control parameters and storage arrays
    theta_tol = ctrl_p['theta_tol_factor'] * np.pi
    a_stop, theta_0, epsilon_fb = 0, 0, 0
    theta_0_arr, epsilon_fb_arr = [0], [0]
    
    # Data storage for analysis and plotting
    P_array, amp_arr, theta_arr = [0], [], [0]
    
    # Variables for cycle detection and control logic
    theta_peak = []
    number_of_cycles = 0
    delta = sim_p['delta'] / dt
    Delta = sim_p['Delta'] / dt
    j_critical = -100  # Sentinel value to track the last critical point
    P_pos_avg, P_neg_avg = 0, 0
    delta_neg, j_neg = 0, 0
    Area_total = 0
    pulse_ratio = sim_p['pos_neg_pulse_ratio']  # Ratio of positive to negative pulse duration

    # --- 2. Main Simulation Loop ---
    for j in range(np.size(t_eval) - 1):
        # Pre-DBS phase
        if j <= int(DBS_time / dt):
            u, u_dot, d, amp, theta, _, _ = phase_est(u, u_dot, d, dt, global_X[-1])
            P = 0
            P_array.append(P)
            amp_arr.append(amp)
            theta_arr.append(theta)
            if j == int(DBS_time / dt):
                a_stop = sim_p['a_stop_factor'] * np.mean(amp_arr)
                print(f"Pre-DBS phase completed. Target amplitude a_stop = {a_stop:.4f}")
        
        # DBS phase where control is active
        else:
            P = 0
            u, u_dot, d, amp, theta, x_cap, y_cap = phase_est(u, u_dot, d, dt, global_X[-1])
            
            # Control logic based on phase window
            # j_critical is used to track the last critical point where abs(theta - theta_0) < theta_tol is True.
            # If it is -100, it means no critical point has been found yet. -100 is a sentinel value.
            flag = False
            if abs(theta - theta_0) < theta_tol:
                flag = True
                if j_critical == -100:
                    j_critical = j
            if flag == False:
                j_critical = -100
            if flag == True:
                P = control_input(amp, theta, theta_0, epsilon_fb)
                # Apply stimulation with on/off periods. Stimulation is applied only for a fixed amount of time in one go.
                # It is pulsated stimulation, with pulse width delta. After the pulse, there is a pause of Delta.
                if (j - j_critical) % (delta + Delta) >= delta:
                    P = 0
            
            # Charge balancing logic. 
            # Here we check if the last peak was below pi, and if so, we calculate the average P for the positive half-cycle.
            # We kept negative pulses equal to a factor of the positive pulses. P_pos/P_neg = pulse_ratio.
            # As we want to keep the charge balanced, this means |Area_pos| =  |Area_neg|.
            # Area = Pulse*width, so delta_neg = |Area_total| / (P_neg_avg * dt).
            if theta_arr[-1] < np.pi and theta >= np.pi and len(theta_peak) > 0:
                Area_pos = 0
                delta_pos = 0
                for s in range(theta_peak[-1], j, 1):
                    Area_pos += P_array[s] * dt
                    if P_array[s] != 0:
                        delta_pos += 1
                P_pos_avg = Area_pos / (delta_pos * dt + 1e-7)
                P_neg_avg = -P_pos_avg/pulse_ratio
                delta_neg = abs(Area_total / (P_neg_avg * dt + 1e-7))
                j_neg = j + delta_neg

            # So after finding the negative pulse height and width(time), we passed it to the control input till stimulation ends. 
            if theta_arr[-1] > np.pi and j <= j_neg:
                P = P_neg_avg

            Area_total += P * dt
            P_array.append(P)
            amp_arr.append(amp)
            theta_arr.append(theta)

        # Cycle detection and adaptive parameter updates
        # If the last three theta values form a peak, we consider it a cycle completion.
        # Adaptive control parameters are updated once a cycle.
        if len(theta_arr) > 2 and theta_arr[-1] < theta_arr[-2] and theta_arr[-2] > theta_arr[-3]:
            number_of_cycles += 1
            theta_peak.append(j - 1)
            if j >= int(DBS_time / dt):
                a_bar = abar(theta_peak, amp_arr)
                theta_0, epsilon_fb = ATSP(a_stop, a_bar, theta_0, epsilon_fb)
                theta_0_arr.append(theta_0)
                epsilon_fb_arr.append(epsilon_fb)

        # Update the state of every oscillator
        for i in range(num_oscillators):
            state = initial_conditions[i]
            dxdt, dydt, dzdt = derivative(state, global_states[0], P, psi=ross_p['psi'], k=ross_p['k'])
            x = euler_method(dxdt, state[0], dt)
            y = euler_method(dydt, state[1], dt)
            z = euler_method(dzdt, state[2], dt)
            initial_conditions[i] = [float(x), float(y), float(z)]

        # Update the global mean-field state
        global_states = np.mean(initial_conditions, axis=0)
        global_X = np.append(global_X, global_states[0])
        global_Y = np.append(global_Y, global_states[1])
        global_Z = np.append(global_Z, global_states[2])

    # --- 3. Final Plots ---
    print("Simulation finished. Generating final plots.")
    plt.figure(figsize=(18, 4))
    plt.plot(t_eval, global_X, label='X')
    plt.plot(t_eval, P_array, label='P')
    plt.xlabel('Time')
    plt.ylabel('X and P')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(epsilon_fb_arr, label='Epsilon_fb')
    plt.plot(theta_0_arr, label='Theta_0')
    plt.xlabel('Number of Cycles')
    plt.ylabel('Adaptive Parameters')
    plt.grid(True)
    plt.legend()
    plt.show()

    return P_array, theta_arr, epsilon_fb_arr, theta_0_arr, global_X, global_Y, global_Z



# --- Main execution block ---
if __name__ == "__main__":
    # Define simulation parameters
    NUM_OSCILLATORS = sim_p['num_oscillators']
    TOTAL_TIME = sim_p['total_time']
    dt = sim_p['dt']
    
    time_array = np.arange(0, TOTAL_TIME, dt)
    
    # Run the main simulation
    P_array, theta_arr, epsilon_fb_arr, theta_0_arr, global_X, global_Y, global_Z = main(num_oscillators=NUM_OSCILLATORS, t_eval=time_array)

    print("Execution complete.")