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
