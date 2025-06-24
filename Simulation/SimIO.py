import json
import numpy as np
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Any
from Simulation.Parameters import *

class CustomEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    


def write_sim_settings(filename, note=None, **params):
    """
    Serializes provided dataclass objects into a JSON file.
    Used to write the settings used from Geometry, Mechanics, Phase field and the simulation 
    into JSon file 

    Parameters:
        filename (str): The path to the output file.
        **params: Keyword arguments where each value is expected to be a dataclass.
        
    # TODO Add a example call
    """
    data = {}
    if note:
        data["Description"] = note
    for key, value in params.items():
        if is_dataclass(value):
            data[key] = asdict(value)
        else:
            raise TypeError(f"Expected dataclass for '{key}', got {type(value).__name__}")
        
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, cls=CustomEncoder)

    print("Parameters have been parsed.")