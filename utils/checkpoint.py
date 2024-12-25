import json
import os
from utils.serialization import convert_to_serializable

def save_checkpoint(current_index, results, args_adls_output, checkpoint=1):
    checkpoint_data = {"current_index": current_index, "results": convert_to_serializable(results)}
    checkpoint_file = f"{args_adls_output}/checkpoint_{checkpoint}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)

def load_checkpoint(args_adls_output, checkpoint=1):
    checkpoint_file = f"{args_adls_output}/checkpoint_{checkpoint}.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
        return checkpoint_data["current_index"], checkpoint_data["results"]
    return 0, {}

