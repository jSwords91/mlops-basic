from typing import Dict, Any
import yaml

def read_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config