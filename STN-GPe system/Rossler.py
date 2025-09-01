import numpy as np
import matplotlib.pyplot as plt
import yaml

# --- Load Configuration from YAML file ---
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
ross_p = config['rossler_params']

# --- Function Definitions ---

def euler_method(dfdt, f, dt):
    """Performs a single forward Euler integration step."""
    return f + dfdt * dt

def derivative(state, X, P=0, psi=ross_p['psi'], k=ross_p['k']):
    """Calculates the derivatives for a single Rössler oscillator in the network."""
    x, y, z = state

    omega = ross_p['omega']
    a = ross_p['a']
    b = ross_p['b']
    c = ross_p['c']
    d = ross_p['d']
    I_ext = ross_p['I_ext']

    # Rossler Equation.
    dxdt = -omega * y - z + k * 0.05 * (d - X) + (0.5 - k) * (0.5 * X - x) + np.cos(psi) * P
    dydt = omega * x + a * y + np.sin(psi) * P
    dzdt = b + z * (x - c) + I_ext
    
    return dxdt, dydt, dzdt

