"""
Provides constants
"""

import os

# Logging Paths
LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)
LOG_FILES_PATH = LOG_PATH + "/files"
METRICS_DICT_PATH = LOG_PATH + "/metrics_dict"
LOG_PICTURES_PATH = LOG_PATH + "/pictures"
