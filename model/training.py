import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import numpy as np
from itertools import product
from datetime import datetime
import time
from utils.checkpoint import save_checkpoint, load_checkpoint, convert_to_serializable
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

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['X'])
        labels = batch['y']
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['X'])
    labels = batch['y']
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))
    return loss

@jax.jit
def convergence_measure(losses, max_val=10, n=4, variance_threshold=0.01):
    gap = 0.1
    n = len(losses) // 20
    losses = jnp.asarray(losses, dtype=jnp.float32)
    losses = jnp.where(jnp.isfinite(losses), losses, max_val)
    initial_loss = losses[0] + 1e-8
    normalized_losses = losses / initial_loss
    normalized_losses = jnp.minimum(normalized_losses, max_val)
    early_losses = normalized_losses[:n]
    recent_losses = normalized_losses[-n:]
    early_mean = jnp.mean(early_losses)
    recent_mean = jnp.mean(recent_losses)
    recent_variance = jnp.mean((recent_losses - recent_mean) ** 2)
    cut_off_ = 0.4
    converged = (recent_mean + gap < early_mean) & (recent_variance < variance_threshold) & (recent_mean < cut_off_)
    sum_loss = jnp.where(converged, jnp.sum(normalized_losses), -jnp.sum((max_val - normalized_losses) / (max_val - 1)))
    cap_n = len(normalized_losses)
    measure = jnp.where(
        converged,
        (((1 + (cap_n - 1) * cut_off_) - jnp.sum(normalized_losses)) / ((cap_n - 1) * cut_off_)) ** 0.5,
        -1 * (((1 + (cap_n - 1) * cut_off_) - jnp.sum(normalized_losses)) / (1 + (cap_n - 1) * (max_val - cut_off_))) ** 0.5
    )
    return measure, sum_loss

def train_model(state, train_ds, epochs=3, batch_size=32, max_loss=1e4):
    num_batches = len(train_ds['X']) // batch_size
    losses = []
    for epoch in range(epochs):
        start_time = time.time()
        perm = np.random.permutation(len(train_ds['X']))
        train_ds_shuffled = {
            'X': train_ds['X'][perm],
            'y': train_ds['y'][perm]
        }
        for i in range(num_batches):
            batch = {
                'X': train_ds_shuffled['X'][i * batch_size:(i + 1) * batch_size],
                'y': train_ds_shuffled['y'][i * batch_size:(i + 1) * batch_size]
            }
            state, loss = train_step(state, batch)
            if loss > max_loss:
                print(f"Stopping early due to divergence. Loss: {loss:.4f}")
                convergence, sum_loss = convergence_measure(losses)
                return state, convergence, sum_loss
            if i % max(1, num_batches // 512) == 0:
                losses.append(float(loss))
        end_time = time.time()
    convergence, sum_loss = convergence_measure(losses)
    return state, convergence, sum_loss

def train_and_evaluate_learning_rates(
    learning_rate_pairs, train_ds, char2idx, idx2char, seq_length, prompt="To be or not to be ",
    vocab_size=96, model_dim=96, num_heads=3, num_layers=2,
    epochs=2, batch_size=32, max_length=96, temperature=0.2,
    generate_text_before_after=True, current_index=0, results={}, checkpoint_ver=1, args_adls_output=''
):
    current_index, results = load_checkpoint(args_adls_output, checkpoint=checkpoint_ver)
    for idx, (t_lr, fc_lr) in enumerate(learning_rate_pairs):
        if idx < current_index:
            continue
        print(f'Processing ... {idx + 1} out of {len(learning_rate_pairs)}')
        rng = jax.random.PRNGKey(0)
        model = SimpleTransformer(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        learning_rates = [
            fc_lr,
            t_lr,
            fc_lr,
            t_lr,
            fc_lr,
            fc_lr
        ]
        state = create_train_state(rng, model, learning_rates, seq_length)
        state, convergence, sum_loss = train_model(
            state, train_ds, epochs=epochs, batch_size=batch_size
        )
        key = f"t_lr_{t_lr:.12f}_fc_lr_{fc_lr:.12f}"
        results[key] = {
            'transformer_lr': t_lr,
            'fc_lr': fc_lr,
            'sum_loss': sum_loss,
            'convergence_measure': convergence
        }
        current_index = idx
    print(len(results))
    return results

def save_results_to_json(results, args_adls_output, filename_prefix='results'):
    serializable_results = convert_to_serializable(results)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{args_adls_output}/{filename_prefix}_{current_time}.json"
    with open(filename, 'w') as json_file:
        json.dump(serializable_results, json_file, indent=4)
    print(f"Results saved to {filename}")
