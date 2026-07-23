import jax
import jax.numpy as jnp
import jax.random as jr

def init_som_lattice(rng, grid_h, grid_w, dim):
    return jr.normal(rng, (grid_h * grid_w, dim)) * 0.02

def som_indices(grid_h, grid_w):
    ys, xs = jnp.meshgrid(jnp.arange(grid_h), jnp.arange(grid_w), indexing="ij")
    return jnp.stack([ys.ravel(), xs.ravel()], axis=1).astype(jnp.float32)

def neighborhood_mask(winner_idx, lattice_coords, sigma):
    winner_coord = lattice_coords[winner_idx]
    dists = jnp.linalg.norm(lattice_coords - winner_coord[None, :], axis=1)
    return jnp.exp(-dists ** 2 / (2.0 * sigma ** 2))

def som_attention(queries, lattice, lattice_coords, sigma):
    sims = jnp.dot(queries, lattice.T)
    winners = jnp.argmax(sims, axis=1)
    def one_token(q_idx):
        w = winners[q_idx]
        mask = neighborhood_mask(w, lattice_coords, sigma)
        logits = sims[q_idx] * mask
        weights = jax.nn.softmax(logits)
        return jnp.dot(weights, lattice)
    return jax.vmap(one_token)(jnp.arange(queries.shape[0]))

def update_lattice(queries, lattice, lattice_coords, lr, sigma):
    sims = jnp.dot(queries, lattice.T)
    winners = jnp.argmax(sims, axis=1)
    def update_for_token(q_idx):
        w = winners[q_idx]
        mask = neighborhood_mask(w, lattice_coords, sigma)
        diff = queries[q_idx][None, :] - lattice
        return lr * mask[:, None] * diff
    deltas = jax.vmap(update_for_token)(jnp.arange(queries.shape[0]))
    return lattice + deltas.sum(axis=0)

class SOMFormerLayer:
    def __init__(self, rng, dim, grid_h, grid_w, n_heads=4):
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_heads = n_heads
        head_dim = dim // n_heads
        rngs = jr.split(rng, 5)
        self.W_q = jr.normal(rngs[0], (dim, dim)) * 0.02
        self.W_k = jr.normal(rngs[1], (dim, dim)) * 0.02
        self.W_v = jr.normal(rngs[2], (dim, dim)) * 0.02
        self.W_o = jr.normal(rngs[3], (dim, dim)) * 0.02
        self.lattices = [init_som_lattice(rngs[4 + i], grid_h, grid_w, head_dim) for i in range(n_heads)]
        self.lattice_coords = som_indices(grid_h, grid_w)

    def __call__(self, x, sigma):
        seq_len = x.shape[0]
        Q = jnp.dot(x, self.W_q)
        V = jnp.dot(x, self.W_v)
        Q = Q.reshape(seq_len, self.n_heads, -1)
        V = V.reshape(seq_len, self.n_heads, -1)
        outputs = []
        for h in range(self.n_heads):
            attended = som_attention(Q[:, h, :], self.lattices[h], self.lattice_coords, sigma)
            outputs.append(attended)
        out = jnp.stack(outputs, axis=1).reshape(seq_len, -1)
        return jnp.dot(out, self.W_o)

if __name__ == "__main__":
    rng = jr.PRNGKey(42)
    layer = SOMFormerLayer(rng, dim=64, grid_h=4, grid_w=4, n_heads=4)
    x = jr.normal(rng, (10, 64))
    sigma = 2.0
    out = layer(x, sigma)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("SOMFormer layer smoke test passed!")
