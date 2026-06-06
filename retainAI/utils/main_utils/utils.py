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