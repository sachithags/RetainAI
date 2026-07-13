import yaml
from retainAI.exception.exception import RetainAIException
import sys, os
from retainAI.logging.logger import logging
import pandas as pd
import numpy as np
import dill
import pickle

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise RetainAIException(e, sys) from e
    
def write_yaml_file(file_path:str, content:dict, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise RetainAIException(e, sys) from e
    

def save_numpy_array_data(file_path:str, array:np.array):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise RetainAIException(e, sys) from e
    

def save_object(file_path:str, obj:object) -> None:
    try:
        logging.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise RetainAIException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Load a .npy file and return the numpy array."""
    return np.load(file_path)