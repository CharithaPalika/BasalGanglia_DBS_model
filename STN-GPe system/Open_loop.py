import numpy as np
import matplotlib.pyplot as plt
import yaml
from Rossler import euler_method, derivative

# --- Load Configuration from YAML file ---
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
ross_p = config['rossler_params']
sim_p = config['open_loop_simulation_params']



def open_loop_control(t, dbs_frequency, P_amp, pos_width, neg_width, dt):
    """Generates a periodic, biphasic pulse train for open-loop stimulation.
    The pulse cycle consists of: +P_amp -> gap -> -P_amp -> gap.
    pos_width: Duration of the positive pulse.
    neg_width: Duration of the negative pulse.
    pos_width + neg_width + gaps = T, where T is the period of the pulse train.
    This implies pos_width + neg_width <= T"""

    T = 100/dbs_frequency # Period of the pulse train
    pos_gap = (T - pos_width - neg_width)/2
    neg_gap = (T - pos_width - neg_width)/2
    P = 0
    # Based on the current time t, determine the control signal P.
    if t%((1/dt)*(pos_width+pos_gap+neg_width+neg_gap)) < (1/dt)*pos_width:
        P = P_amp
    elif t%((1/dt)*(pos_width+pos_gap+neg_width+neg_gap)) < (1/dt)*(pos_width + pos_gap):
        P = 0
    elif t%((1/dt)*(pos_width+pos_gap+neg_width+neg_gap)) < (1/dt)*(pos_width + pos_gap + neg_width):
        P = -P_amp
    else:
        P = 0
    
    return P



def main(num_oscillators, t_eval, dbs_frequency, pos_width = sim_p['pos_width'], neg_width = sim_p['neg_width'], p_amp = sim_p['p_amp']):
    """Main function to run the open-loop simulation."""
    print("Starting open-loop DBS simulation...")

    # --- 1. Initialization ---
    dt = sim_p['dt']
    DBS_time = sim_p['DBS_start_time']
    initial_conditions = np.zeros((num_oscillators, 3))
    initial_conditions[:, 0] = np.random.normal(0, 0.001, num_oscillators)

    global_states = np.mean(initial_conditions, axis=0)
    global_X = np.array([global_states[0]])
    global_Y = np.array([global_states[1]])
    global_Z = np.array([global_states[2]])
    P_array = [0]

    # --- 2. Main Simulation Loop ---
    for j in range(np.size(t_eval) - 1):

        # Determine the control signal P
        if j <= int(DBS_time / dt):
            # Pre-DBS phase: No stimulation
            P = 0
        else:
            # Open-loop stimulation phase
            P = open_loop_control(j, dbs_frequency, p_amp, pos_width, neg_width, dt)
        P_array.append(P)

        # Update the state of every oscillator in the network
        for i in range(num_oscillators):
            state = initial_conditions[i]
            dxdt, dydt, dzdt = derivative(state, global_states[0], P, psi=ross_p['psi'], k=ross_p['k'])
            x = euler_method(dxdt, state[0], dt)
            y = euler_method(dydt, state[1], dt)
            z = euler_method(dzdt, state[2], dt)
            initial_conditions[i] = [float(x), float(y), float(z)]

        # Update the global mean-field state for the next time step
        global_states = np.mean(initial_conditions, axis=0)
        global_X = np.append(global_X, global_states[0])
        global_Y = np.append(global_Y, global_states[1])
        global_Z = np.append(global_Z, global_states[2])

    # --- 3. Final Plots ---
    print("Simulation finished. Generating final plots.")
    plt.figure(figsize=(18, 4))
    plt.plot(t_eval, global_X, label='Mean-field X')
    plt.plot(t_eval, P_array, label='Stimulation P')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

    return P_array, global_X, global_Y, global_Z


# --- Main execution block ---
if __name__ == "__main__":
    # Define simulation parameters
    NUM_OSCILLATORS = sim_p['num_oscillators'] # Number of Rössler oscillators in the network
    TOTAL_TIME = sim_p['total_time']  # Total simulation time in seconds
    DBS_FREQUENCY = sim_p['input_freq']  # Frequency of the DBS pulse train
    
    # Create the time array for the simulation
    time_array = np.arange(0, TOTAL_TIME, sim_p['dt'])
    
    print(f"Running simulation for {TOTAL_TIME} seconds with a DBS frequency of {DBS_FREQUENCY} Hz.")
    
    for freq in DBS_FREQUENCY:
        print(f"Running simulation for DBS frequency: {freq} Hz")
        P_array, global_X, global_Y, global_Z = main(
                                                    num_oscillators=NUM_OSCILLATORS,
                                                    t_eval=time_array,
                                                    dbs_frequency=freq,
                                                    pos_width=sim_p['pos_width'],
                                                    neg_width=sim_p['neg_width'],
                                                    p_amp=sim_p['p_amp']
                                                )

    print("Execution complete.")