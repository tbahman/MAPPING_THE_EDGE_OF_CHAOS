import jax.numpy as jnp
import flax.linen as nn

class TransformerLayer(nn.Module):
    model_dim: int
    num_heads: int

    def setup(self):
        self.self_attention = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.model_dim)
        self.layer_norm1 = nn.LayerNorm()
        self.feed_forward = nn.Sequential([nn.Dense(self.model_dim * 2), nn.relu, nn.Dense(self.model_dim)])
        self.layer_norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None):
        x_res = x + self.self_attention(self.layer_norm1(x), mask)
        return x_res + self.feed_forward(self.layer_norm2(x_res))

class SimpleTransformer(nn.Module):
    vocab_size: int
    model_dim: int
    num_heads: int
    num_layers: int

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.model_dim)
        self.layers = [TransformerLayer(self.model_dim, self.num_heads) for _ in range(self.num_layers)]
        self.fc = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x) + sinusoidal_positional_encoding(x.shape[1], self.model_dim)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

