# AirFormer:Anchor-Slice Physics Attention for Airflow Prediction

## Task

Given past 5 velocity snapshots on a 3D airfoil mesh with up to 100000 points, 
the task is to predict the future 5 velocity snapshots. Each point has a 3D position
and a 3D velocity vector per timestep, and airfoil surface points, that are subject to 
no-slip boundary condition. 

---

## Method

We decouple spatial reasoning from per-point prediction. Instead of operating attention on all 100k mesh points, we sample 4096 **anchor points**, process them with a Physics Attention backbone, and interpolate the result back to the full mesh.

**Input encoding.** For each point we concatenate a Fourier positional encoding (Tancik et al., *NeurIPS 2020*), the flattened 5-step velocity history, and a binary surface flag, then project to 128 dimensions with a small MLP. This per-point feature doubles as a skip connection later.

**Anchor selection.** We draw `M = 4096` anchors in a 50/50 split between surface and field points motivated by stratified sampling principles (Lai et al., *CVPR 2022*; Qian et al., *NeurIPS 2022*) and the observation that boundary-layer regions dominate prediction error. Each anchor is initialised by mean-pooling its 16 nearest mesh-point features using a single `cKDTree` query.

**Backbone.** We run 12 pre-norm Transolver blocks (Wu et al., *ICML 2024*) on the anchor tokens. Each block soft-routes anchors into 128 learnable physical-state slices, performs attention between slice tokens (16 heads, 16 dims/head), and broadcasts back giving us linear-time global attention at a manageable token count.

**Readout.** Every mesh point looks up its 8 nearest anchors (one more `cKDTree` call), pools their features through an inverse-distance softmax with a learnable temperature, concatenates the result with its skip feature, and passes through a fusion MLP. Five independent decoder heads: each with a learned step bias produce velocity deltas that we add to the last observed velocity and zero out at airfoil points for exact no-slip enforcement.

---

## Model specifications

| Hyperparameter                  | Value  |
|---------------------------------|--------|
| Per-point feature dim           | 128     |
| Anchor feature dim              | 256    |
| Anchor count `M`                | 4096  |
| Physics Attention blocks        | 12     |
| Attention heads                 | 16     |
| Slices per block `G`            | 128    |
| Surface anchor fraction         | 0.50   |
| Anchor seeding neighbours       | 16     |
| Per-point readout neighbours    | 8      |
| Fourier frequency bands         | 8      |
| Dropout                         | 0.05   |
| Input timesteps                 | 5      |
| Output timesteps                | 5      |
| **Total parameters**            | **~9.7 M** |

---

## Training

- **Dataset**: 100,000-point 3D airfoil meshes at full resolution, 5 past
  and 5 future velocity snapshots per sample. 
- **Optimiser**: AdamW, base learning
  rate `2e-4`, weight decay `1e-4`.
- **Schedule**: 2-epoch linear warm-up, cosine decay to `1e-6` over 10
  epochs.
- **Gradient clipping**: max-norm 1.0.
- **Mixed precision**: AMP (float16).
- **Batch size**: 4.

### Loss

The training loss is a combination of the mean-squared error, along-with other point-wise L1 and L2 losses, and also a temporal loss:

```
L = w_mse * L_mse
  + w_metric * L_metric
  + w_l1 * L_l1
  + w_mag * L_magnitude
  + w_temp * L_temporal
```

| Term            |  Weight  |
|-----------------|----------|
| `L_mse`         |  `1.0`   |
| `L_metric`      |  `1.0`   |
| `L_l1`          |  `0.1`   |
| `L_magnitude`   |  `0.1`   |
| `L_temporal`    |  `0.5`   |

---

## Results

Trained on a single NVIDIA T4 GPU (16 GB) for 10 epochs with AMP enabled,
689 training samples and 121 validation samples. Total training time was
approximately 60 minutes.

| Metric                          | Value     |
|---------------------------------|-----------|
| Best validation L2              | 1.2053    |
| Training epochs                 | 10        |
| Best epoch                      | 10        |
| Total parameters                | 9,676,880 |
| Time per epoch                  | ~363s     |

### Epoch-Level Summary

| Epoch | Train Loss | Train L2 | Val Loss | Val L2     |
|-------|------------|----------|----------|------------|
| 1     | 13.85      | 1.6679   | 12.15    | 1.5452     |
| 2     | 12.42      | 1.6096   | 10.21    | 1.4120     |
| 3     | 10.50      | 1.4507   | 9.07     | 1.3090     |
| 4     | 9.88       | 1.3962   | 8.72     | 1.2792     |
| 5     | 11.07      | 1.3727   | 9.84     | 1.2981     |
| 6     | 10.80      | 1.3513   | 9.68     | 1.2604     |
| 7     | 10.58      | 1.3319   | 9.40     | 1.2267     |
| 8     | 10.43      | 1.3203   | 9.25     | 1.2165     |
| 9     | 10.32      | 1.3099   | 9.16     | 1.2074     |
| 10    | 10.23      | 1.3019   | **9.13** | **1.2053** |

---

## Disclaimer

Parts of this code-base and documentation have been written with the assistance of LLMs. 

---

## References

- Wu H., Luo H., Wang H., Wang J., Long M. **Transolver: A Fast Transformer
  Solver for PDEs on General Geometries.** *ICML 2024.*
- Lai X., Liu J., Jiang L. et al. **Stratified Transformer for 3D Point
  Cloud Segmentation.** *CVPR 2022.*
- Qian G., Li Y., Peng H. et al. **PointNeXt: Revisiting PointNet++ with
  Improved Training and Scaling Strategies.** *NeurIPS 2022.*
- Tancik M., Srinivasan P.P., Mildenhall B. et al. **Fourier Features Let
  Networks Learn High Frequency Functions in Low Dimensional Domains.**
  *NeurIPS 2020.*