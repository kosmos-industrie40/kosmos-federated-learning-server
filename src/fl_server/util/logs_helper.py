"""
Provides helper methods for logging
"""
import os
import json
from pathlib import Path
from typing import Dict


def store_dict(
    output_directory_path: Path,
    dictionary: Dict[str, Dict[str, Dict[str, float]]],
    experiment_name: str,
):
    """
    Save dictionary as json file
    """
    if not os.path.exists(output_directory_path):
        Path(output_directory_path).mkdir(parents=True, exist_ok=True)
    output_file_path = output_directory_path.joinpath(experiment_name + ".json")
    with output_file_path.open("w") as text_file:
        json.dump(dictionary, text_file, indent=4)
