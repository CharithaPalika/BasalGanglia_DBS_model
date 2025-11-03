# Basal Ganglia & STN-GPe Simulation Workspace
The code is made available for review purposes
This repository contains two main modules:

1. **Decision Making (Basal Ganglia Model):**  
   Simulates the Iowa Gambling Task (IGT) and Nonstationary Bandits using the computational model of the Basal Ganglia (BG) under normal, Parkinson's Disease (PD), and Deep Brain Stimulation (DBS) conditions.

2. **STN-GPe System (Rössler Oscillator Network):**  
   Models the subthalamic nucleus (STN) – globus pallidus externus (GPe) system as a network of Rössler oscillators, capturing normal, PD, and DBS dynamics. Includes open-loop and closed-loop DBS protocols.

---

## Directory Structure

### Decision Making (BG Model)
- **Jupyter Notebooks:**  
  - `IGT_Normal.ipynb`, `IGT_PD.ipynb`, `Nonstationary_bandits_Normal.ipynb`, `Nonstationary_bandits_PD.ipynb`: Run simulations and visualize results.
- **BasalGanglia/**  
  - `BGNetwork.py`: BG neural network model  
  - `rossler_network.py`: Rössler oscillator network for STN input simulation  
  - `train.py`: Training and simulation routines  
  - `params.yaml`: Model parameters
- **Environments/**  
  - `igt.py`: IGT environment implementation  
  - `NonStationaryBandits.py`: Nonstationary bandit environment
- **hyperparams/**  
  - `IGT_hyperparams.yaml`: Hyperparameters for IGT  
  - `NonStationaryBandits_hyperparams.yaml`: Hyperparameters for bandits
- **utils/**  
  - `utils.py`: Utility functions (e.g., YAML loading)

### STN-GPe System
- **Open_loop.py:** Simulates open-loop DBS with periodic biphasic pulses
- **Closed_loop.py:** Simulates closed-loop DBS using adaptive feedback control
- **Rossler.py:** Rössler network model of the STN-GPe system
- **config.yaml:** Configuration file for simulation parameters

---

## Getting Started

### Install Dependencies
```sh
pip install numpy torch matplotlib seaborn tqdm pyyaml pandas scipy
```

### Run Decision Making Simulations
- Open a notebook (e.g., `IGT_Normal.ipynb`) in Jupyter or VS Code.
- Adjust hyperparameters in `hyperparams/IGT_hyperparams.yaml` as needed.
- Execute cells to run experiments and visualize results.

### Run STN-GPe Simulations
- Edit `config.yaml` to adjust model or DBS stimulation parameters.
- Run open-loop simulation:
  ```sh
  python Open_loop.py
  ```
- Run closed-loop simulation:
  ```sh
  python Closed_loop.py
  ```
- View the plots generated after each simulation.

---

## Main Components

- **IGTEnv:** Simulates the Iowa Gambling Task environment (`Environments/igt.py`)
- **BGNetwork:** PyTorch neural network model of the Basal Ganglia (`BasalGanglia/BGNetwork.py`)
- **train:** Main training loop for running simulations (`BasalGanglia/train.py`)
- **Rossler Network:** Models STN-GPe dynamics (`Rossler.py`)
- **DBS Protocols:** Open-loop and closed-loop stimulation (`Open_loop.py`, `Closed_loop.py`)
---

For questions or issues, please open an issue or contact
