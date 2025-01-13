import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import numpy as np
from itertools import product
from datetime import datetime
import time
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.preprocessing import convert_to_serializable
from model.transformer import SimpleTransformer
from utils.signal_handler import register_signal_handler

def create_train_state(rng, model, learning_rates, seq_length):
    _, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.ones([1, seq_length], dtype=jnp.int32))['params']
    txs = {
        'embedding': optax.adam(learning_rates[0]),
        'attention': optax.adam(learning_rates[1]),
        'dense': optax.adam(learning_rates[2]),
        'layer_norm_attention': optax.adam(learning_rates[3]),
        'layer_norm_dense': optax.adam(learning_rates[4]),
        'fc': optax.adam(learning_rates[5]),
    }

    def label_params(params):
        def match_fn(param_name):
            if param_name.startswith('embedding/'):
                return 'embedding'
            elif 'self_attention' in param_name:
                return 'attention'
            elif 'feed_forward' in param_name:
                return 'dense'
            elif 'layer_norm1' in param_name:
                return 'layer_norm_attention'
            elif 'layer_norm2' in param_name or 'layer_norm_final' in param_name:
                return 'layer_norm_dense'
            elif param_name.startswith('fc/'):
                return 'fc'
            else:
                return 'dense'
        flat_params = flax.traverse_util.flatten_dict(params)
        labels = {path: match_fn('/'.join(path)) for path in flat_params}
        return flax.traverse_util.unflatten_dict(labels)

    tx = optax.multi_transform(txs, label_params(params))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

