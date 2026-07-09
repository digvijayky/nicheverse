# Architecture and method

## Problem

A Xenium panel resolves a few hundred genes at single transcript resolution across millions of cells per slide. The standard unsupervised workflow (Leiden over PCA) ignores spatial context: a tumor cell next to a T cell and the same tumor cell deep inside a sterile region get the same label even though the local tissue is biologically distinct. We wanted a representation that learns recurrent transcriptional states **and** recurrent multicellular niches in a single pass, so that downstream analyses (compositional shifts, niche enrichment, ligand receptor) can operate on a discrete vocabulary.

## Architecture

For each cell i with feature vector x_i in R^G (G genes) and physical coordinates (x_i, y_i) in microns, we build a paired neighborhood feature h_i by aggregating the k nearest neighbor cells within the same Xenium run. The k-NN graph is constructed per sample using a ball tree on physical coordinates so that no edges cross slides.

Default aggregation is inverse-distance weighted mean:

    h_i = sum_{j in N_k(i)} w_ij * x_j,   w_ij ~ 1 / (d_ij + epsilon)

We expose `mean` and `max` aggregations as alternatives but use `weighted_mean` in the manuscript runs.

Two separate MLP encoders operate on the cell features and on the concatenated `(x_i, h_i)`:

    z_cell    = CellEncoder(x_i)              in R^{d_c}
    z_neigh   = NeighEncoder([x_i ; h_i])     in R^{d_n}

Each is quantized against its own learned codebook (K_c entries of dimension d_c, K_n entries of dimension d_n) using EMA updates, k-means++ initialization, dead-code reset every 50 batches, and an entropy regularizer that keeps the assignment distribution close to uniform.

A multi-head cross attention block lets the cell representation attend to its quantized niche before reconstruction:

    z_cell_attended = CrossAttn(Q=z_cell, K=V=Proj(z_neigh_q))
    z_cell_final    = z_cell_q + 0.5 * z_cell_attended

The two decoders mirror the encoders and reconstruct `x_i` and `[x_i ; h_i]` respectively.

Loss:

    L = L_cell_recon + L_cell_commit + L_neigh_recon + L_neigh_commit + L_diversity

with commitment cost 0.25 by default. The diversity term penalizes low-entropy code usage.

## Why two codebooks instead of one

A single codebook over `[x_i ; h_i]` collapses transcriptional state and niche identity into one factor, which makes downstream compositional analysis ambiguous: a shift in code frequency could be a shift in cell composition, a shift in spatial organization, or both. With two codebooks, the cell code is interpretable as a transcriptional state (Cell type / state) and the niche code as a multicellular composition (tissue niche). Each can be relabeled independently against literature, and shifts can be decomposed into "what changed in cells" versus "what changed in tissue context".

## Why cross attention

The cell code is what most downstream consumers care about: it is the analog of a single-cell type label. Without cross attention, the cell encoder is blind to its niche, so a cancer cell in a tertiary lymphoid structure gets the same code as one buried in stroma. The 0.5 residual mixing keeps the cell code dominated by the cell's own transcripts while letting niche context modulate borderline assignments.

## VQ design choices

EMA updates (decay 0.99) keep the codebook smooth and avoid the gradient noise problems of standard VQ. K-means++ initialization on the first batch lets the codebook start at well separated points rather than uniform random in `[-1, 1]`. The dead code reset replaces any code that drops below 1% of average usage with a perturbed sample from the current batch, which prevents the persistent codebook collapse that the original VQ-VAE paper struggles with. The entropy regularizer pulls toward uniform assignment with a small weight (1.0) so it doesn't override the reconstruction loss but does prevent a few codes from absorbing all the mass.

## Defaults used in the Cancer Cell submission

```
cell_num_embeddings        = 256
cell_embedding_dim         = 64
neighborhood_num_embeddings = 32
neighborhood_embedding_dim = 256
hidden_dims                = (256, 128)
commitment_cost            = 0.25
use_cross_attention        = True
k_neighbors                = 20
neighborhood_aggregation   = weighted_mean
batch_size                 = 2048
learning_rate              = 3e-4
num_epochs                 = 300
optimizer                  = Adam
scheduler                  = ReduceLROnPlateau (factor 0.5, patience 5)
preprocessing              = scanpy normalize_total + log1p
seed                       = 49
```

## Choosing K_c and K_n

Rule of thumb: pick K_c large enough to give a few "spare" codes (10 to 30% unused) so that the dead code reset has room to escape collapses, but small enough that you can manually annotate each code. We have used K_c = 128 (kidney injury panel) and K_c = 256 (RCC + BrM, 366 genes). K_n is typically smaller because tissue niches are coarser than cell states: K_n = 16 to 32 works for most cohorts.

If many codes go unused after 300 epochs, lower K_c. If perplexity is close to K_c (uniform usage), the model is using all of capacity; try increasing K_c by 50%.

## Computational complexity

Training cost is dominated by the per-sample k-NN graph at startup (O(n log n) per sample) and the per-epoch forward + backward pass (O(n) per epoch). For the 173 sample cohort: k-NN graph 4 minutes, training 25 minutes per epoch on one A100. Inference cost is identical to one epoch's forward pass.
