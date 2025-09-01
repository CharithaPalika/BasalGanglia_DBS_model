import yaml
import numpy as np

def load_yaml(yaml_file):
    ''' 
    Function to load yaml file

    Args:
        yaml_file (str): File path 
    Returns:
        data (dict): Returns loaded yaml file as dictionary
    '''
    with open(yaml_file,'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, filename):
    with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
