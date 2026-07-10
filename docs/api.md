# API reference

Every public symbol below is documented inline with its parameters, tensor
shapes, returns, and role in the NICHEVERSE model. The reconstruction model,
its quantizers, and encoders live in {mod}`nicheverse.models`; the spatial
dataset, readers, trainer, losses, prediction, and annotation utilities are
exported at the top level of {mod}`nicheverse`.

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

## Data

Reading imaging spatial transcriptomics into an ``AnnData`` and building the
spatial neighbor graph the model consumes.

```{eval-rst}
.. autoclass:: nicheverse.SpatialDataset
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autofunction:: nicheverse.read_spatial
.. autofunction:: nicheverse.read_xenium
.. autofunction:: nicheverse.read_xenium_cohort
.. autofunction:: nicheverse.load_xenium_run
.. autofunction:: nicheverse.load_xenium_cohort
.. autofunction:: nicheverse.spatial_neighbors
.. autofunction:: nicheverse.attach_codes
.. autofunction:: nicheverse.attach_codes_to_adata
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
per-code marker and DEG evidence and, optionally, primary literature.

```{eval-rst}
.. autoclass:: nicheverse.annotate.AnnotationConfig
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autofunction:: nicheverse.annotate_codes
.. autofunction:: nicheverse.code_evidence
.. autofunction:: nicheverse.annotate.annotate_niches
.. autofunction:: nicheverse.annotate.niche_evidence
.. autofunction:: nicheverse.annotate.cluster_codes
.. autofunction:: nicheverse.annotate.attach_labels
.. autofunction:: nicheverse.annotate.code_dotplot
.. autofunction:: nicheverse.annotate.pubmed_search
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
