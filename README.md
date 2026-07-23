# PC-SOMFormer

Predictive Coding + Self-Organizing Map Transformer (PC-SOMFormer) built with JAX.

## Requirements

- Python 3.9+
- JAX with NumPy backend (CPU-only is fine)

## Setup

```bash
git clone https://github.com/martok9803/pc-somformer.git
cd pc-somformer
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib
```

## Running

```bash
# Smoke test the SOMFormer layer
python somformer.py

# Smoke test the full PC-SOMFormer
python pc_somformer.py
```

### Expected output for pc_somformer.py

```
Input shape:  (10, 64)
Output shape: (10, 64)
Prediction error: 0.xxxx
PC-SOMFormer smoke test passed!
```

## File Overview

| File | Description |
|------|-------------|
| `somformer.py` | SOMFormer layer with SOM-based attention and lattice updates |
| `pc_somformer.py` | Full model wrapping SOMFormer layers with Predictive Coding (predict-observe-revise) |