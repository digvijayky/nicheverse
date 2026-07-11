# Architecture and method

## Problem

A Xenium panel resolves a few hundred genes at single transcript resolution across millions of cells per slide. The standard unsupervised workflow (Leiden over PCA) ignores spatial context: a tumor cell next to a T cell and the same tumor cell deep inside a sterile region get the same label even though the local tissue is biologically distinct. We wanted a representation that learns recurrent transcriptional states **and** recurrent multicellular niches in a single pass, so that downstream analyses (compositional shifts, niche enrichment, ligand receptor) can operate on a discrete vocabulary.

## Architecture

For each cell i with feature vector x_i in R^G (G genes) and physical coordinates (x_i, y_i) in microns, we build a paired neighborhood feature h_i by aggregating the k nearest neighbor cells within the same Xenium run. The k-NN graph is constructed per sample using a ball tree on physical coordinates so that no edges cross slides.

Default aggregation is inverse-distance weighted mean:

    h_i = sum_{j in N_k(i)} w_ij * x_j,   w_ij ~ 1 / (d_ij + epsilon)

We expose `mean` and `max` aggregations as alternatives but use `weighted_mean` in the manuscript runs.

Two separate encoders operate on the cell features and on the concatenated `(x_i, h_i)`. The default backbone is `mlp_deep`, a SwiGLU pre-norm residual MLP (no per-gene numerical embedding), chosen because it gives the healthiest raw codebook on sparse Xenium counts:

    z_cell    = CellEncoder(x_i)              in R^{d_c}
    z_neigh   = NeighEncoder([x_i ; h_i])     in R^{d_n}

Each is quantized against its own learned codebook (K_c entries of dimension d_c, K_n entries of dimension d_n) using EMA updates, k-means++ initialization, dead-code reset, and an entropy regularizer that keeps the assignment distribution close to uniform. The EMA codebook is updated by the EMA rule rather than the optimizer and is frozen from gradient descent.

A multi-head cross attention block lets the cell representation attend to its quantized niche before reconstruction:

    z_cell_attended = CrossAttn(Q=z_cell, K=V=Proj(z_neigh_q))
    z_cell_final    = z_cell_q + 0.5 * z_cell_attended

The two decoders mirror the encoders and reconstruct the cell counts and the neighborhood composition respectively.

Loss:

    L = L_cell_recon + L_cell_commit + L_neigh_recon + L_neigh_commit + L_diversity

with commitment cost 0.25 by default. By default the cell reconstruction term is a negative-binomial NLL on the raw counts (a scVI-style library size taken from the observed total count) plus a Bernoulli detection hurdle weighted by `detection_weight` (default 0.5); there is no MSE on the cell branch. The niche reconstruction term is composition MSE plus a Dirichlet-multinomial on the count-scale aggregated-neighbor composition. These count-native defaults require raw INTEGER counts in `adata.X`; the pure-MSE path (MSE on log1p for the cell branch, composition MSE for the niche branch) is recoverable with `cell_recon="mse"` and `niche_recon="mse"`. The diversity term penalizes low-entropy code usage.

## Why two codebooks instead of one

A single codebook over `[x_i ; h_i]` collapses transcriptional state and niche identity into one factor, which makes downstream compositional analysis ambiguous: a shift in code frequency could be a shift in cell composition, a shift in spatial organization, or both. With two codebooks, the cell code is interpretable as a transcriptional state (Cell type / state) and the niche code as a multicellular composition (tissue niche). Each can be relabeled independently against literature, and shifts can be decomposed into "what changed in cells" versus "what changed in tissue context".

## Why cross attention

The cell code is what most downstream consumers care about: it is the analog of a single-cell type label. Without cross attention, the cell encoder is blind to its niche, so a cancer cell in a tertiary lymphoid structure gets the same code as one buried in stroma. The 0.5 residual mixing keeps the cell code dominated by the cell's own transcripts while letting niche context modulate borderline assignments.

## VQ design choices

EMA updates (decay 0.99) keep the codebook smooth and avoid the gradient noise problems of standard VQ; the EMA codebook is frozen from the optimizer so only the EMA rule moves it. K-means++ initialization on the first batch lets the codebook start at well separated points rather than uniform random in `[-1, 1]`. The dead code reset replaces any code that drops below 1% of average usage with a perturbed sample from the current batch, which prevents the persistent codebook collapse that the original VQ-VAE paper struggles with. The entropy regularizer pulls toward uniform assignment with a small weight (1.0) so it doesn't override the reconstruction loss but does prevent a few codes from absorbing all the mass.

## Defaults used in the Cancer Cell submission

```
encoder_type               = mlp_deep
quantizer_type             = vq
cell_num_embeddings        = 256
cell_embedding_dim         = 64
neighborhood_num_embeddings = 32
neighborhood_embedding_dim = 256
hidden_dims                = (256, 128)
commitment_cost            = 0.25
use_cross_attention        = True (one-directional, cell reads niche)
cross_attention_heads      = 4
cross_attention_weight     = 0.5
cell_recon                 = nb (+ detection hurdle, detection_weight=0.5)
niche_recon                = mse_dirmult
spatial_graph              = knn_radius (radius 50 microns)
k_neighbors                = 20
neighborhood_aggregation   = weighted_mean
batch_size                 = 32768
learning_rate              = 3e-4
num_epochs                 = 300
optimizer                  = AdamW (decoupled selective weight decay 0.01)
scheduler                  = ReduceLROnPlateau (factor 0.5, patience 5)
preprocessing              = scanpy normalize_total + log1p (encoder input); raw counts kept as the NB / Dirichlet-multinomial target
seed                       = 9
```

## Choosing K_c and K_n

Rule of thumb: pick K_c large enough to give a few "spare" codes (10 to 30% unused) so that the dead code reset has room to escape collapses, but small enough that you can manually annotate each code. We have used K_c = 128 (kidney injury panel) and K_c = 256 (RCC + BrM, 366 genes). K_n is typically smaller because tissue niches are coarser than cell states: K_n = 16 to 32 works for most cohorts.

If many codes go unused after 300 epochs, lower K_c. If perplexity is close to K_c (uniform usage), the model is using all of capacity; try increasing K_c by 50%.

## Computational complexity

Training cost is dominated by the per-sample k-NN graph at startup (O(n log n) per sample) and the per-epoch forward + backward pass (O(n) per epoch). For the 173 sample cohort at the b32k reference configuration, the per-sample graph is built once before training and each epoch's forward plus backward pass runs in about 17 seconds on one A100 (about 330,000 cells per second), so 300 epochs complete in about 1.4 hours (1:25:48 measured) at a peak GPU memory of about 43 GB. Inference cost is close to one epoch's forward pass.
