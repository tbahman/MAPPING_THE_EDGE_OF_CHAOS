import json
import os
import flax
import numpy as np
import jax.numpy as jnp

def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, jax.Array):
        return np.array(obj).tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

def save_checkpoint(current_index, results, args_adls_output, checkpoint=1):
    checkpoint_data = {
        "current_index": current_index,
        "results": convert_to_serializable(results)
    }
    checkpoint_file = f"{args_adls_output}/checkpoint_{checkpoint}.json"
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"Checkpoint version {checkpoint} saved at index {current_index}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")

def load_checkpoint(args_adls_output, checkpoint=1):
    checkpoint_file = f"{args_adls_output}/checkpoint_{checkpoint}.json"
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"Checkpoint version {checkpoint} loaded. Resuming from index {checkpoint_data['current_index']}")
            return checkpoint_data['current_index'], checkpoint_data['results']
        else:
            print("No checkpoint found. Starting from the beginning.")
            return 0, {}
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return 0, {}

