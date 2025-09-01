# Simulating Pathological Oscillations and DBS Control in STN–GPe Networks
This repository models the subthalamic nucleus (STN) – globus pallidus externus (GPe) system as a network of Rössler oscillators,capturing the dynamics characteristic of Basal ganglia (BG) circuits under normal and Parkinsonian conditions. It provides code to test different Deep Brain Stimulation (DBS) protocols on STN-GPe system , implementing both open-loop and closed-loop paradigms. The closed-loop protocol is based on Rosenblum (2020). The framework enables exploration of how DBS strategies modulate the pathological oscillations in STN-GPe system.
## Contents

- **Open_loop.py**  
  Simulates open-loop DBS with periodic biphasic pulses.

- **Closed_loop.py**  
  Simulates closed-loop DBS using adaptive feedback control based on the oscillatory activity within STN-GPe system.

- **Rossler.py**  
  Contains the Rössler netowrk model of STN-GPe system.

- **config.yaml**  
  Configuration file for all simulation parameters.

## How to Use

1. **Install Python 3 and required packages:**
   ```
   pip install numpy matplotlib pyyaml
   ```

2. **Edit `config.yaml`** to adjust any model or DBS stimulation parameters if needed.

3. **Run a simulation:**
   - For open-loop:  
     ```
     python Open_loop.py
     ```
   - For closed-loop:  
     ```
     python Closed_loop.py
     ```

4. **View the plots** generated after each simulation.

## Notes

- All parameters (oscillator, stimulation, control) are set in `config.yaml`.
- Results are displayed as plots at the end of each run. 

## References

- Rosenblum, M. (2020). Controlling collective synchrony in oscillatory ensembles by precisely timed pulses. Chaos: An Interdisciplinary Journal of Nonlinear Science, 30(9).
---

For questions, please contact the repository