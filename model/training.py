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

