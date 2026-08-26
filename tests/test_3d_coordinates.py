"""Volumetric (3D) coordinate support.

Imaging platforms increasingly emit true 3D positions (3D MERFISH of whole
embryos, for instance). Coordinates used to be truncated to the first two
columns, which builds the neighbourhood graph on a projection and mixes cells
from different depths. These tests pin the corrected behaviour.
"""
import numpy as np
import pytest
import anndata as ad

from nicheverse import SpatialDataset


def _adata(n=120, g=12, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.poisson(2.0, (n, g)).astype(np.float32))
    a.obs["sample_id"] = ["s0"] * (n // 2) + ["s1"] * (n - n // 2)
    a.obsm["spatial"] = rng.uniform(0, 100, (n, dim))
    a.var_names = [f"g{i}" for i in range(g)]
    return a


def test_3d_coordinates_are_kept_not_truncated():
    a = _adata(dim=3)
    ds = SpatialDataset.from_anndata(a, k_neighbors=5)
    assert ds.spatial_coords.shape[1] == 3
    np.testing.assert_allclose(ds.spatial_coords, np.asarray(a.obsm["spatial"]))


def test_2d_still_works():
    ds = SpatialDataset.from_anndata(_adata(dim=2), k_neighbors=5)
    assert ds.spatial_coords.shape[1] == 2


def test_four_column_coordinates_raise_rather_than_silently_project():
    a = _adata(dim=4)
    with pytest.raises(ValueError, match="must be"):
        SpatialDataset.from_anndata(a, k_neighbors=5)


def test_depth_actually_changes_the_neighbourhood():
    """Two slabs share the same x,y but sit 1000 units apart in z. If z were
    dropped they would be mutual neighbours and the two slabs would aggregate to
    the same neighbourhood profile. Keeping z must separate them."""
    rng = np.random.default_rng(1)
    m = 40
    xy = rng.uniform(0, 10, (m, 2))
    n = 2 * m
    X = np.zeros((n, 6), np.float32)
    X[:m, :3] = rng.poisson(6.0, (m, 3))      # slab A expresses the first genes
    X[m:, 3:] = rng.poisson(6.0, (m, 3))      # slab B the last genes
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(6)]
    a.obs["sample_id"] = ["s0"] * n
    xyz = np.hstack([np.vstack([xy, xy]),
                     np.r_[np.zeros(m), np.full(m, 1000.0)][:, None]])
    a.obsm["spatial"] = xyz

    ds3 = SpatialDataset.from_anndata(a.copy(), k_neighbors=5, spatial_graph="knn")
    flat = a.copy(); flat.obsm["spatial"] = xyz[:, :2]
    ds2 = SpatialDataset.from_anndata(flat, k_neighbors=5, spatial_graph="knn")

    def nbr_half(ds, i):
        _, nbr, _ = ds[i]
        return np.asarray(nbr)[6:]           # aggregated-neighbour half

    # in 3D a slab-A cell sees only slab-A genes; flattened, it sees both
    a3 = np.stack([nbr_half(ds3, i) for i in range(m)])
    a2 = np.stack([nbr_half(ds2, i) for i in range(m)])
    assert a3[:, 3:].sum() < a3[:, :3].sum() * 0.05
    assert a2[:, 3:].sum() > a3[:, 3:].sum()


@pytest.mark.parametrize("graph", ["knn", "knn_radius", "radius", "delaunay"])
def test_graph_modes_run_in_3d(graph):
    kw = {"radius": 40.0} if graph in ("knn_radius", "radius") else {}
    ds = SpatialDataset.from_anndata(_adata(dim=3), k_neighbors=5,
                                     spatial_graph=graph, **kw)
    assert len(ds) == 120
    cell, nbr, _ = ds[0]
    assert cell.shape[0] == 12 and nbr.shape[0] == 24
