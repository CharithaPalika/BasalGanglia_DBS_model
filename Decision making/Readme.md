# IGT Basal Ganglia Simulation Workspace

This repository contains code for simulating the Iowa Gambling Task (IGT) using computational models of the Basal Ganglia, including closed-loop and open-loop Deep Brain Stimulation (DBS) paradigms. The workspace supports experiments with normal, Parkinson's Disease (PD), and DBS conditions.

## Directory Structure

- **IGT_Closed_DBS_del_lim_loop.ipynb**  
- **IGT_Open_DBS_high_del_lim_loop.ipynb**  
- **IGT_Open_DBS_low_del_lim_loop.ipynb**  
- **IGT_PD_del_lim_loop.ipynb**  
- **IGT_PD.ipynb**  
- **IGT_STD_DBS_del_lim_loop.ipynb**  
- **IGT_Normal.ipynb**  
  Jupyter notebooks for running different IGT simulation scenarios.

- **BasalGanglia/**  
  - [`BGNetwork.py`](BasalGanglia/BGNetwork.py): Implements the Basal Ganglia neural network model.  
  - [`rossler_network.py`](BasalGanglia/rossler_network.py): Rossler oscillator network for STN input simulation.  
  - [`train.py`](BasalGanglia/train.py): Training and simulation routines.  
  - [`params.yaml`](BasalGanglia/params.yaml): Model parameters.

- **Environments/**  
  - [`igt.py`](Environments/igt.py): IGT environment implementation.  
  - [`NonStationaryBandits.py`](Environments/NonStationaryBandits.py): Nonstationary bandit environment.

- **hyperparams/**  
  - [`IGT_hyperparams.yaml`](hyperparams/IGT_hyperparams.yaml): Hyperparameters for experiments.

- **utils/**  
  - [`utils.py`](utils/utils.py): Utility functions (e.g., YAML loading).

## Getting Started

1. **Install Dependencies**  
   - Python 3.8+  
   - Required packages: `numpy`, `torch`, `matplotlib`, `seaborn`, `tqdm`, `yaml`, `pandas`, `scipy`
   - Install with pip:
     ```sh
     pip install numpy torch matplotlib seaborn tqdm pyyaml pandas scipy
     ```

2. **Run Simulations**  
   - Open any notebook (e.g., `IGT_Normal.ipynb`) in Jupyter or VS Code.
   - Adjust hyperparameters in [`hyperparams/IGT_hyperparams.yaml`](hyperparams/IGT_hyperparams.yaml) as needed.
   - Execute cells to run experiments and visualize results.

3. **Modify Models**  
   - Change network architecture in [`BasalGanglia/BGNetwork.py`](BasalGanglia/BGNetwork.py).
   - Update training logic in [`BasalGanglia/train.py`](BasalGanglia/train.py).

## Main Components

- **IGTEnv** ([`Environments/igt.py`](Environments/igt.py)):  
  Simulates the Iowa Gambling Task environment with reward/loss schedules.

- **BGNetwork** ([`BasalGanglia/BGNetwork.py`](BasalGanglia/BGNetwork.py)):  
  PyTorch neural network model of the Basal Ganglia.

- **train** ([`BasalGanglia/train.py`](BasalGanglia/train.py)):  
  Main training loop for running simulations.

## Citation

If you use this codebase for research, please cite the original publication or contact the authors.

## License

This project is for academic use only. See LICENSE file for details.

---

For questions or issues, please open an issue or contact the repository