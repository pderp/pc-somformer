import jax
import jax.numpy as jnp
import jax.random as jr
from somformer import SOMFormerLayer, update_lattice

class PCSOMFormer:
    """SOMFormer with Predictive Coding Forward Chaining (PREDICT-OBSERVE-REVISE)."""
    def __init__(self, rng, dim, grid_h, grid_w, n_heads=4, n_layers=2, lr=0.01, sigma=2.0, sigma_decay=0.99):
        self.dim = dim
        self.n_layers = n_layers
        self.lr = lr
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        rngs = jr.split(rng, n_layers)
        self.layers = [SOMFormerLayer(rngs[i], dim, grid_h, grid_w, n_heads) for i in range(n_layers)]
        self.pred_weights = [jr.normal(jr.split(rngs[i])[1], (dim, dim)) * 0.02 for i in range(n_layers)]

    def forward_pass(self, x, sigma):
        activations = [x]
        for layer in self.layers:
            x = layer(x, sigma)
            activations.append(x)
        return activations

    def top_down_predict(self, upper_act, layer_idx):
        return jnp.dot(upper_act, self.pred_weights[layer_idx])

    def compute_error(self, predicted, observed):
        return observed - predicted

    def revise_activations(self, observed, predicted, error, lr_revise=0.1):
        return observed + lr_revise * error

    def pc_cycle(self, x, training=False):
        sigma = self.sigma
        activations = self.forward_pass(x, sigma)
        errors = []
        revised_activations = [activations[0]]
        for i in range(self.n_layers):
            upper = activations[i + 2] if i + 2 < len(activations) else activations[-1]
            predicted = self.top_down_predict(upper, i)
            observed = activations[i + 1]
            error = self.compute_error(predicted, observed)
            errors.append(error)
            revised = self.revise_activations(observed, predicted, error)
            revised_activations.append(revised)
        revised_x = revised_activations[1] if len(revised_activations) > 1 else x
        sigma = sigma * self.sigma_decay
        revised_output = self.forward_pass(revised_x, sigma)[-1]
        if training:
            for i, layer in enumerate(self.layers):
                for h in range(layer.n_heads):
                    head_dim = layer.dim // layer.n_heads
                    acts = revised_activations[i + 1].reshape(-1, head_dim)
                    layer.lattices[h] = update_lattice(acts, layer.lattices[h], layer.lattice_coords, self.lr, sigma)
        total_error = jnp.sum(jnp.stack([jnp.mean(jnp.abs(e)) for e in errors]))
        return revised_output, total_error, errors

    def __call__(self, x, training=False):
        output, error, _ = self.pc_cycle(x, training)
        return output, error


if __name__ == "__main__":
    rng = jr.PRNGKey(42)
    model = PCSOMFormer(rng, dim=64, grid_h=4, grid_w=4, n_heads=4, n_layers=2)
    x = jr.normal(rng, (10, 64))
    out, err = model(x, training=False)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Prediction error: {err:.4f}")
    print("PC-SOMFormer smoke test passed!")
