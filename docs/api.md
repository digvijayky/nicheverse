# API reference

Every public symbol below is documented inline with its parameters, tensor
shapes, returns, and role in the NICHEVERSE model. The reconstruction model,
its quantizers, and encoders live in {mod}`nicheverse.models`; the spatial
dataset, readers, trainer, losses, prediction, and annotation utilities are
exported at the top level of {mod}`nicheverse`.

The [Mathematical formulation](#mathematical-formulation) section documents the
core classes in the style of the PyTorch `torch.nn` docs: a signature block, a
one-paragraph description, a **Parameters** list, a **Shape** section, and the
governing equations. The autodoc entries below it mirror the code exactly.

(mathematical-formulation)=
## Mathematical formulation

Notation. A mini-batch holds $B$ cells. The cell branch input is the
log1p-normalized expression $x \in \mathbb{R}^{B \times G}$ over $G$ genes; the
neighborhood branch input is $[\,x \;;\; h\,] \in \mathbb{R}^{B \times 2G}$, the
concatenation of a cell's own expression and its aggregated neighbor expression
$h$. Encoders map these to latents $z_e^{\text{cell}} \in \mathbb{R}^{B \times
d_c}$ and $z_e^{\text{niche}} \in \mathbb{R}^{B \times d_n}$, each snapped to its
codebook. Symbols: $K$ codebook size, $D$ code dimension, $\operatorname{sg}[\cdot]$
the stop-gradient operator.

**Total training objective.** The trainer minimizes the sum of the two branch
losses, each a reconstruction likelihood plus its VQ (commitment + diversity)
term, weighted by `cell_weight` and `neighborhood_weight` (both $1.0$ by default):

$$
\mathcal{L} = \underbrace{w_{\text{cell}}\bigl(\mathcal{L}_{\text{recon}}^{\text{cell}} + \mathcal{L}_{\text{VQ}}^{\text{cell}}\bigr)}_{\text{cell branch}} + \underbrace{w_{\text{niche}}\bigl(\mathcal{L}_{\text{recon}}^{\text{niche}} + \mathcal{L}_{\text{VQ}}^{\text{niche}}\bigr)}_{\text{niche branch}},
$$

where $\mathcal{L}_{\text{recon}}^{\text{cell}}$ is the negative-binomial +
detection-hurdle count likelihood, $\mathcal{L}_{\text{recon}}^{\text{niche}}$ is
the composition MSE + Dirichlet-multinomial term, and each
$\mathcal{L}_{\text{VQ}}$ is the commitment (+ diversity) loss of its quantizer
(all defined below). Optional spatial-coherence regularizers are added only when
`spatial_loss_weight > 0` (off by default).

### `HierarchicalVQVAE(config)`

Two paired codebooks coupled by cross-attention. The cell encoder produces a cell
latent that is quantized to a cell-state code; the neighborhood encoder produces a
niche latent that is quantized to a niche code; a one-directional gated
cross-attention lets the cell latent read its niche before both branches are
decoded and reconstructed.

**Parameters**

- **config** (*ModelConfig*) - input dimensionality, codebook sizes ($K_c, K_n$),
  embedding dimensions ($d_c, d_n$), encoder / quantizer selection, cross-attention
  flag and residual weight, and the reconstruction-loss modes. See
  {class}`~nicheverse.ModelConfig`.

**Shape**

- Cell input: $(B, G)$ with $G = \texttt{input\_dim}$.
- Neighborhood input: $(B, 2G)$.
- Cell reconstruction: $(B, G)$; niche reconstruction: $(B, 2G)$.
- Code indices: $(B, 1)$ each; VQ losses and perplexities are scalars.

The quantized cell latent $z_q^{\text{cell}}$ is conditioned on the quantized niche
latent $z_q^{\text{niche}}$ by a single multi-head attention read, mixed back with a
**fixed** residual weight $\lambda = \texttt{cross\_attention\_weight}$ (default
$0.5$; this is a constant residual gate, not a learned sigmoid gate):

$$
z_q^{\text{cell,final}} = z_q^{\text{cell}} + \lambda \,\operatorname{Attn}\!\bigl(Q = z_q^{\text{cell}},\; K = V = W_p\, z_q^{\text{niche}}\bigr),
\qquad
\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\!\Bigl(\tfrac{QK^{\top}}{\sqrt{d_c}}\Bigr) V,
$$

with $W_p$ the learned niche-to-cell projection (`neighborhood_projection`) and the
attention a `torch.nn.MultiheadAttention` over a single-token sequence per cell
(so the softmax is over one key and reduces to $V$, i.e. the read returns the
projected niche vector; the head count is the largest divisor of $d_c$ at most
`cross_attention_heads`, default $4$). This is one-directional: the niche latent is
never updated from the cell (the niche code stays a pure tissue descriptor). The
decoders $g_{\text{cell}}, g_{\text{niche}}$ then map $z_q^{\text{cell,final}}$ and
$z_q^{\text{niche}}$ back to the input spaces.

### `VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, ...)`

The default quantizer (`quantizer_type="vq"`). Given an encoded latent it assigns
each row to the nearest of $K$ codebook entries $\{e_k\}_{k=1}^{K}$, returns the
quantized vector with straight-through gradients, and refreshes the codebook by
exponential moving average rather than by the optimizer.

**Parameters**

- **num_embeddings** (*int*) - codebook size $K$.
- **embedding_dim** (*int*) - code dimensionality $D$.
- **commitment_cost** (*float*) - commitment weight $\beta$ (default $0.25$ via
  `ModelConfig.commitment_cost`).
- **ema_decay** (*float*, default $0.99$) - EMA decay $\gamma$ for the cluster
  sizes and embedding sums.
- **diversity_weight** (*float*, default $1.0$) - weight of the entropy diversity
  term; $0$ disables it.
- **distance_metric** (*str*, default `"l2"`) - `"l2"` (squared Euclidean) or
  `"cosine"`.

**Shape**

- Input: $(B, D, T)$ channels-first ($T=1$ in the hierarchical model).
- Output: quantized $(B, D, T)$; `encoding_indices` $(B\,T, 1)$; `loss`,
  `perplexity` scalars.

Codebook lookup assigns each latent $z_e$ to the nearest code by squared Euclidean
distance:

$$
k^\star = \arg\min_{j \in \{1,\dots,K\}} \bigl\lVert z_e - e_j \bigr\rVert_2^2, \qquad z_q = e_{k^\star}.
$$

Because $\arg\min$ has no gradient, the encoder is trained through a
straight-through estimator, $z_q \leftarrow z_e + \operatorname{sg}[\,z_q - z_e\,]$,
which copies the decoder gradient at $z_q$ back onto $z_e$. With the EMA codebook the
optimizer sees only the commitment term (the codebook is frozen from gradients),

$$
\mathcal{L}_{\text{VQ}} = \beta \,\bigl\lVert \operatorname{sg}[\,z_q\,] - z_e \bigr\rVert_2^2 + w_{\text{div}}\bigl(\log K - H(\bar p)\bigr),
$$

where $\bar p$ is the batch-mean soft assignment $\operatorname{softmax}(-\text{dist}/\tau)$
and $H$ is Shannon entropy (the diversity term pushes usage toward uniform). Without
EMA the codebook term $\lVert z_q - \operatorname{sg}[z_e]\rVert_2^2$ is added instead.
The codebook itself is updated by EMA of the per-code assignment counts $n_k$ and
the assigned-vector sums $m_k = \sum_{i : k^\star_i = k} z_{e,i}$:

$$
N_k \leftarrow \gamma N_k + (1-\gamma)\, n_k, \qquad
M_k \leftarrow \gamma M_k + (1-\gamma)\, m_k, \qquad
e_k \leftarrow \frac{M_k}{N_k},
$$

with a Laplace smoothing of $N_k$ (van den Oord 2017; Razavi 2019). Every
`dead_code_reset_interval` steps, codes whose usage falls below a fraction of their
fair share are reseeded from random batch rows.

### `mlp_deep` encoder (`DeepMLPEncoder`)

The default cell / neighborhood backbone (`encoder_type="mlp_deep"`): a linear input
projection into a residual stream of width $d = \texttt{hidden}[0]$, a stack of
pre-norm SwiGLU residual blocks, an output LayerNorm, and a linear projection to the
latent. No per-gene numerical embedding (which degenerates on sparse Xenium counts).

**Parameters**

- **in_dim, out_dim** (*int*) - input gene count $G$ (or $2G$ for the neighborhood
  branch) and output latent dimension.
- **hidden** (*Sequence[int]*) - residual-stream width is `hidden[0]`.
- **dropout** (*float*, default $0.1$) - dropout inside the SwiGLU blocks.
- **n_blocks** (*int*, optional) - number of residual blocks (default
  $\max(4, \texttt{len(hidden)})$).
- **ffn_mult** (*float*, default $2.0$) - SwiGLU hidden expansion factor.

**Shape**

- Input: $(B, \texttt{in\_dim})$. Output: $(B, \texttt{out\_dim})$.

Each block updates the residual stream $z$ with a pre-norm gated feed-forward:

$$
z \leftarrow z + \operatorname{Drop}\!\Bigl(W_o\bigl[(W_1\,\operatorname{LN}(z)) \odot \operatorname{SiLU}(W_2\,\operatorname{LN}(z))\bigr]\Bigr),
$$

where $\odot$ is the elementwise product (the SwiGLU gate) and $\operatorname{LN}$ is
LayerNorm. The output is $\texttt{proj\_out}(\operatorname{LN}(z))$.

### Reconstruction losses

The default cell branch (`cell_recon="nb"`) is a negative-binomial count
likelihood on the raw integer counts plus a detection hurdle; there is no MSE on the
cell branch. Let $x \in \mathbb{Z}_{\ge 0}^{B \times G}$ be the raw counts,
$\ell_i = \sum_g x_{ig}$ the observed library size (scVI convention), $p =
\operatorname{softmax}(\text{decoder logits})$ the per-gene proportion, and $\mu = \ell
\odot p$ the NB mean. With the learned per-gene inverse-dispersion $\theta =
\exp(\texttt{cell\_log\_theta})$, the NB negative log-likelihood is

$$
\mathrm{NLL}_{\text{NB}}(x) = -\sum_{g}\Bigl[ \log\tfrac{\Gamma(x_g + \theta_g)}{\Gamma(\theta_g)\,\Gamma(x_g+1)} + \theta_g \log\tfrac{\theta_g}{\theta_g + \mu_g} + x_g \log\tfrac{\mu_g}{\theta_g + \mu_g}\Bigr].
$$

The detection hurdle is a per-gene Bernoulli / BCE on the $\mathbf{1}[x>0]$ mask,
reusing the decoder output as the detection logits $\eta$:

$$
\mathrm{BCE}(x) = -\sum_g \bigl[\, \mathbf{1}[x_g > 0]\log \sigma(\eta_g) + \mathbf{1}[x_g = 0]\log(1 - \sigma(\eta_g)) \,\bigr].
$$

Both terms are reduced to per-gene means (divided by $G$) so they are comparable,
giving the default cell reconstruction loss

$$
\mathcal{L}_{\text{recon}}^{\text{cell}} = \tfrac{1}{G}\,\mathrm{NLL}_{\text{NB}}(x) + w_{\text{det}}\cdot \tfrac{1}{G}\,\mathrm{BCE}(x), \qquad w_{\text{det}} = 0.5,
$$

with the detection weight $w_{\text{det}} = \texttt{detection\_weight}$. The `"mse"`
mode instead recovers the released Gaussian (MSE-on-log1p) term
$\lVert \hat x - x\rVert_2^2 / (BG)$; `"poisson"` uses the equidispersed count NLL
(the NB with $\theta \to \infty$); `"both"` sums $w_{\text{mse}}\,\mathrm{MSE} +
w_{\text{nb}}\,\tfrac{1}{G}\mathrm{NLL}_{\text{NB}}$.

The default niche branch (`niche_recon="mse_dirmult"`) sums a composition MSE on the
log1p niche vector and a Dirichlet-multinomial NLL on the count-scale
aggregated-neighbor composition $c$ (row sum $N = \sum_g c_g$). With the mean
composition $p = \operatorname{softmax}(\text{logits})$, a learned precision $\alpha_0
= \sum_g \exp(\texttt{niche\_log\_alpha}_g)$, and Dirichlet parameters $\alpha = p\,
\alpha_0$, the DM negative log-likelihood is

$$
\mathrm{NLL}_{\text{DM}}(c) = -\Bigl[ \log\Gamma(N+1) - \sum_g \log\Gamma(c_g+1) + \log\Gamma(\alpha_0) - \log\Gamma(\alpha_0 + N) + \sum_g \bigl(\log\Gamma(c_g+\alpha_g) - \log\Gamma(\alpha_g)\bigr)\Bigr],
$$

evaluated with `lgamma` so fractional $c$ (the continuous weighted-mean aggregate)
needs no rounding. The niche loss is $w_{\text{mse}}\,\mathrm{MSE} + w_{\text{dm}}\,
\tfrac{1}{G}\mathrm{NLL}_{\text{DM}}$ (both weights default $1.0$).

### `MoleculeSetVQVAE(...)`

The molecule-set variant of {class}`~nicheverse.HierarchicalVQVAE`: the cell branch
is a {class}`~nicheverse.models.MoleculeSetEncoder` over a cell's transcript point
cloud (one token per molecule, $\text{token} = E_{\text{gene}}(g) +
\operatorname{MLP}(\operatorname{Fourier}(dx, dy))$), pooled by
$\operatorname{concat}[\text{masked max},\ \text{masked mean},\ \text{PMA seed}]$
(never PMA alone, which collapses the pooled embedding). It feeds the same cell
codebook, an optional aggregated-kNN neighborhood branch, and the same gated
cross-attention, so the emitted cell / niche code indices match the standard model.

**Parameters**

- **n_genes** (*int*, default $366$) - panel size (also the padding gene index).
- **cell_embedding_dim, cell_num_embeddings** (*int*) - cell code dimension $d_c$
  and count $K_c$ (defaults $64$, $256$).
- **neighborhood_embedding_dim, neighborhood_num_embeddings** (*int*) - niche code
  dimension $d_n$ and count $K_n$ (defaults $256$, $32$).
- **commitment_cost** (*float*, default $0.25$) - VQ commitment weight $\beta$.
- **use_neighborhood, use_cross_attention** (*bool*, default `True`) - enable the
  niche branch and the cell-reads-niche cross-attention.
- **cross_attention_weight** (*float*, default $0.5$) - residual mix $\lambda$.
- **enc_width, enc_isab, enc_heads, enc_inds, enc_freqs** - Set-Transformer encoder
  width, number of ISAB blocks, attention heads, induced points, and Fourier
  frequencies.

**Shape**

- Molecule inputs: `gene` $(B, M)$ long, `coords` $(B, M, 2)$ microns, `mask`
  $(B, M)$ bool; $M$ is the padded molecule count. Neighborhood features (optional):
  $(B, 2G)$.
- Cell reconstruction: $(B, G)$; code indices $(B, 1)$ each.

### `ModelConfig` defaults (b32k production model)

The released RCC/BrM production model (`b32k`) is reproduced by the
{class}`~nicheverse.ModelConfig` defaults below. Where a field's dataclass default
differs from a legacy checkpoint's serialized value, the code default is authoritative
(the loader falls back to the pre-refactor `cell_recon="default"` / `niche_recon="mse"`
/ `detection_weight=0.0` only for checkpoints saved before those fields existed).

| Field | Default | Meaning |
| --- | --- | --- |
| `cell_num_embeddings` × `cell_embedding_dim` | $256 \times 64$ | cell codebook $K_c \times d_c$ |
| `neighborhood_num_embeddings` × `neighborhood_embedding_dim` | $32 \times 256$ | niche codebook $K_n \times d_n$ |
| `commitment_cost` | $0.25$ | VQ commitment weight $\beta$ |
| `encoder_type` | `"mlp_deep"` | SwiGLU pre-norm residual MLP |
| `quantizer_type` | `"vq"` | EMA codebook (default {class}`~nicheverse.VectorQuantizer`) |
| `vq_distance` | `"l2"` | squared-Euclidean code assignment |
| `use_cross_attention` | `True` | cell reads niche |
| `cross_attention_weight` | $0.5$ | residual mix $\lambda$ |
| `cross_attention_heads` | $4$ | attention heads |
| `cell_recon` | `"nb"` | NB count likelihood + detection hurdle |
| `detection_weight` | $0.5$ | hurdle weight $w_{\text{det}}$ |
| `niche_recon` | `"mse_dirmult"` | composition MSE + Dirichlet-multinomial |
| `w_niche_mse`, `w_dirmult` | $1.0$, $1.0$ | niche-loss weights |
| `hidden_dims` | $(256, 128)$ | encoder MLP widths (decoder reversed) |

### `TrainConfig` defaults (b32k trajectory)

The released training trajectory is reproduced by the {class}`~nicheverse.TrainConfig`
defaults below.

| Field | Default | Meaning |
| --- | --- | --- |
| `num_epochs` | $300$ | full passes over the cohort |
| `batch_size` | $32768$ | cells per mini-batch (b32k) |
| `learning_rate` | $3\times10^{-4}$ | Adam base LR (plateau-scheduled) |
| `weight_decay` | $0.01$ | decoupled AdamW (weights only, codebooks excluded) |
| `spatial_graph` | `"knn_radius"` | k-NN capped at `radius` microns, per sample |
| `radius` | $50.0$ | niche-edge cap in microns |
| `k_neighbors` | $20$ | neighbors per cell |
| `neighborhood_aggregation` | `"weighted_mean"` | inverse-distance neighbor pooling |
| `cell_weight`, `neighborhood_weight` | $1.0$, $1.0$ | branch multipliers $w_{\text{cell}}, w_{\text{niche}}$ |
| `lr_schedule` | `"plateau"` | halve LR on val/train plateau (patience $5$) |
| `spatial_loss_weight` | $0.0$ | spatial-coherence regularizer off by default |

## Model

The hierarchical VQ-VAE: a per-cell encoder, a cell codebook, a neighborhood
codebook, and the cross-attention that couples them.

```{eval-rst}
.. autoclass:: nicheverse.ModelConfig
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.HierarchicalVQVAE
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autofunction:: nicheverse.save_checkpoint
.. autofunction:: nicheverse.load_checkpoint
```

## Quantizers

Each quantizer maps a continuous embedding to a discrete code. ``vq`` is the
released default; the others are swappable through ``ModelConfig.quantizer_type``
and the registry.

```{eval-rst}
.. autoclass:: nicheverse.VectorQuantizer
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.FSQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.SoftVQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.RotVQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.QINCoVQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.ProductVQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.ResidualVQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.LFQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.BSQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.ResidualFSQ
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.GroupedResidualVQ
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autofunction:: nicheverse.models.build_quantizer
.. autofunction:: nicheverse.models.register_quantizer
```

## Encoders

Encoder backbones map each cell's gene vector to a latent, chosen with
``ModelConfig.encoder_type`` and configured with ``ModelConfig.encoder_kwargs``.
All share the builder signature ``(in_dim, out_dim, hidden, dropout, **kwargs)``
and map ``(batch, in_dim)`` to ``(batch, out_dim)``.

| ``encoder_type`` | Backbone | ``encoder_kwargs`` |
| --- | --- | --- |
| ``mlp_deep`` (default) | Deep, wide SwiGLU pre-norm residual MLP (no per-gene numerical embedding) | ``ffn_mult``, ``n_blocks`` |
| ``mlp`` | Plain MLP: Linear, BatchNorm, ReLU, Dropout | (none) |
| ``mlp_plr`` | MLP with per-gene periodic numerical embeddings and a SwiGLU trunk | ``n_freq``, ``num_emb_dim``, ``n_blocks``, ``freq_init_sigma``, ``ffn_mult`` |
| ``residual_mlp`` | Pre-activation residual MLP blocks | (none) |
| ``transformer`` | Gene-patch Transformer | ``patch_size``, ``num_heads``, ``num_layers`` |
| ``cnn`` | 1D convolutional network over the gene vector | ``channels``, ``kernel_sizes``, ``use_batch_norm`` |
| ``fast_cnn`` | Lightweight 1D CNN | ``channels``, ``kernel_sizes``, ``use_batch_norm`` |
| ``deep_cnn`` | Deep residual 1D CNN with squeeze-and-excitation and multiscale fusion | ``base_channels``, ``depth_multiplier``, ``n_stages``, ``res_blocks_per_stage``, ``kernel_sizes``, ``use_se``, ``use_multiscale`` |
| ``gnn`` | Graph-attention message passing over gene-patch tokens | ``patch_size``, ``num_layers`` |
| ``diffusion`` | U-Net denoiser encoder | ``time_emb_dim``, ``n_timesteps`` |
| ``dit`` | Diffusion Transformer with AdaLN conditioning | ``patch_size``, ``num_heads``, ``num_layers``, ``time_emb_dim``, ``n_timesteps`` |
| ``set_transformer`` | Set Transformer (ISAB, PMA pooling) over gene-patch tokens | ``patch_size``, ``num_heads``, ``num_inds``, ``num_isab`` |
| ``perceiver_io`` | Perceiver IO latent cross-attention | ``patch_size``, ``num_heads``, ``num_latents``, ``num_self_layers`` |
| ``soft_moe`` | Soft Mixture-of-Experts routing | ``patch_size``, ``num_heads``, ``num_experts``, ``slots_per_expert`` |
| ``ft_transformer`` | FT-Transformer with per-gene feature tokens (compute-prohibitive at cohort scale) | ``num_heads``, ``num_layers`` |

```{eval-rst}
.. autofunction:: nicheverse.models.build_encoder
.. autofunction:: nicheverse.models.register_encoder
```

```{eval-rst}
.. autoclass:: nicheverse.models.ResidualMLP
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.TransformerEncoder
   :members:
   :member-order: bysource
```

## Molecule-set model

An alternative input modality: instead of a per-cell aggregated gene vector, each
cell is encoded as the *set of its transcript molecules* (one token per molecule,
carrying gene identity and subcellular offset) through a permutation-invariant,
mask-aware Set Transformer. :class:`~nicheverse.models.MoleculeSetVQVAE` mirrors
:class:`~nicheverse.HierarchicalVQVAE` (cell codebook, optional neighborhood
codebook, cross-attention fusion) and reuses the released
:class:`~nicheverse.VectorQuantizer`, so the codebook behavior and emitted code
indices are unchanged.

```{eval-rst}
.. autoclass:: nicheverse.models.MoleculeSetEncoder
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.models.MoleculeSetVQVAE
   :members:
   :member-order: bysource
```

## Data

Reading imaging spatial transcriptomics into an ``AnnData`` and building the
spatial neighbor graph the model consumes.

```{eval-rst}
.. autoclass:: nicheverse.SpatialDataset
   :members:
   :member-order: bysource
```

The spatial graph mode is chosen with ``SpatialDataset.from_anndata(...,
spatial_graph=...)`` (mirrored by ``TrainConfig.spatial_graph``): ``knn_radius``
(default, a k-NN graph whose edges are capped at ``radius`` microns, default
``50``), ``knn`` (pure k-NN, no cap), ``radius``, ``delaunay``, ``alpha_complex``,
``gabriel``, and ``rng``. The graph is always built per sample, so cells from
different samples never link. Passing ``transcript_context_key`` concatenates a
segmentation-free molecular field (see :func:`~nicheverse.data.transcript_context`)
onto the counts, doubling the model input dimension.

```{eval-rst}
.. autofunction:: nicheverse.read_spatial
.. autofunction:: nicheverse.read_xenium
.. autofunction:: nicheverse.read_xenium_cohort
.. autofunction:: nicheverse.load_xenium_run
.. autofunction:: nicheverse.load_xenium_cohort
.. autofunction:: nicheverse.spatial_neighbors
.. autofunction:: nicheverse.data.transcript_context
.. autofunction:: nicheverse.attach_codes
.. autofunction:: nicheverse.attach_codes_to_adata
```

The subcellular molecule-set representation (a per-cell transcript point cloud) is
loaded by :class:`~nicheverse.data.MoleculeSetDataset` and consumed by
:class:`~nicheverse.models.MoleculeSetVQVAE`.

```{eval-rst}
.. autoclass:: nicheverse.data.MoleculeSetDataset
   :members:
   :member-order: bysource
```

## Training

```{eval-rst}
.. autoclass:: nicheverse.Trainer
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.TrainConfig
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autofunction:: nicheverse.train_model
.. autofunction:: nicheverse.mae_pretrain
```

## Prediction

```{eval-rst}
.. autofunction:: nicheverse.predict_codes
```

## Losses

Optional spatial-coherence and codebook regularizers added to the
reconstruction objective.

```{eval-rst}
.. automodule:: nicheverse.losses
   :members:
   :member-order: bysource
```

## Annotation

Turn learned codes into labeled cell states and spatial niches, grounded in
per-code marker and DEG evidence and, optionally, primary literature. The
end-to-end harness is {func}`~nicheverse.annotate.annotate_codebook`, which
builds the evidence, proposes labels, gates them against markers and lab rules,
scores calibration, and returns an {class}`~nicheverse.annotate.AnnotationResult`.
The individual stages ({func}`~nicheverse.code_evidence`,
{func}`~nicheverse.annotate_codes`, the verify and evaluate helpers) are also
callable on their own.

### Harness

```{eval-rst}
.. autofunction:: nicheverse.annotate.annotate_codebook
.. autoclass:: nicheverse.annotate.AnnotationResult
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.annotate.AnnotationConfig
   :members:
   :member-order: bysource
```

### Project context and priors

```{eval-rst}
.. autoclass:: nicheverse.annotate.ProjectContext
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.annotate.CellTypePrior
   :members:
   :member-order: bysource
.. autoclass:: nicheverse.annotate.NichePrior
   :members:
   :member-order: bysource
```

### Evidence and labeling

```{eval-rst}
.. autofunction:: nicheverse.annotate_codes
.. autofunction:: nicheverse.code_evidence
.. autofunction:: nicheverse.annotate.annotate_niches
.. autofunction:: nicheverse.annotate.niche_evidence
.. autofunction:: nicheverse.annotate.cluster_codes
.. autofunction:: nicheverse.annotate.code_context
.. autofunction:: nicheverse.annotate.code_groundtruth_concordance
.. autofunction:: nicheverse.annotate.write_evidence_bundle
.. autofunction:: nicheverse.annotate.attach_labels
.. autofunction:: nicheverse.annotate.code_dotplot
```

### Verification and scoring

The verify stage checks proposed labels against the evidence and lab rules; the
evaluate stage scores per-code agreement and calibrates confidence.

```{eval-rst}
.. autofunction:: nicheverse.annotate.gate
.. autofunction:: nicheverse.annotate.marker_presence
.. autofunction:: nicheverse.annotate.check_citations
.. autofunction:: nicheverse.annotate.apply_lab_rules
.. autofunction:: nicheverse.annotate.validate_vocabulary
.. autofunction:: nicheverse.annotate.score_code
.. autofunction:: nicheverse.annotate.calibration
.. autofunction:: nicheverse.annotate.scorecard_table
.. autofunction:: nicheverse.annotate.write_provenance_manifest
```

### Literature and LLM providers

```{eval-rst}
.. autofunction:: nicheverse.annotate.pubmed_search
.. autofunction:: nicheverse.annotate.biorxiv_search
.. autofunction:: nicheverse.annotate.literature_for_markers
.. autofunction:: nicheverse.annotate.call_llm
```

## Plotting

```{eval-rst}
.. automodule:: nicheverse.plotting
   :members:
   :member-order: bysource
```

## Utilities

Determinism, environment capture, and hashing for byte-exact reproduction.

```{eval-rst}
.. autoclass:: nicheverse.Keys
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autofunction:: nicheverse.seed_everything
.. autofunction:: nicheverse.env_snapshot
.. autofunction:: nicheverse.write_env_snapshot
.. autofunction:: nicheverse.sha256_array
.. autofunction:: nicheverse.sha256_file
.. autofunction:: nicheverse.anndata_keys
```
