# creates object for params in config.yaml file. 
import yaml
from types import SimpleNamespace

def dict_to_objects(data):
    """converts a dict into class objects, make it easier to use parameter in the main loop"""

    if not isinstance(data, dict):
        return data

    ns = SimpleNamespace()

    for key, value in data.items():
        # not directly passing value, because it enables the nested attr in yaml files; eg (model: dof: 7 )
        setattr(ns, key, dict_to_objects(value)) 

    return ns

def load_robot_config(path):
    """opens the yaml file and load the parameter into dict"""
    with open(path, 'r') as f:
        parameters_in_config_file = yaml.safe_load(f)
        return dict_to_objects(parameters_in_config_file)


