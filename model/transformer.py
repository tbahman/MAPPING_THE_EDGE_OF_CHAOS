import jax.numpy as jnp
import flax.linen as nn

def sinusoidal_positional_encoding(seq_len, model_dim):
    position = jnp.arange(seq_len)[:, None]
    i = jnp.arange(model_dim)[None, :]
    angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / model_dim)
    angle_rads = position * angle_rates
    sines = jnp.sin(angle_rads[:, 0::2])
    cosines = jnp.cos(angle_rads[:, 1::2])
    pos_encoding = jnp.zeros((seq_len, model_dim))
    pos_encoding = pos_encoding.at[:, 0::2].set(sines)
    pos_encoding = pos_encoding.at[:, 1::2].set(cosines)
    return pos_encoding

class TransformerLayer(nn.Module):
    model_dim: int
    num_heads: int

    def setup(self):
        self.self_attention = nn.SelfAttention(
            num_heads=self.num_heads, 
            dtype=jnp.float32,
            qkv_features=self.model_dim,
            use_bias=False
        )
        self.layer_norm1 = nn.LayerNorm()
        self.feed_forward = nn.Sequential([
            nn.Dense(self.model_dim * 2),
            nn.relu,
            nn.Dense(self.model_dim),
        ])
        self.layer_norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None):
        x_res = x
        x = self.layer_norm1(x)
        x = self.self_attention(x, mask=mask)
        x = x_res + x
        x_res = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x_res + x
        return x

class SimpleTransformer(nn.Module):
    vocab_size: int
    model_dim: int
    num_heads: int
    num_layers: int

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.model_dim)
        self.transformer_layers = [
            TransformerLayer(model_dim=self.model_dim, num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ]
        self.layer_norm_final = nn.LayerNorm()
        self.fc = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        seq_len = x.shape[1]
        pos_encoding = sinusoidal_positional_encoding(seq_len, self.model_dim)
        x += pos_encoding
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, seq_len, seq_len))
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)
        x = self.layer_norm_final(x)
        x = self.fc(x)
        return x
