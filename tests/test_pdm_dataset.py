"""Regression tests for PDM properties reaching real PyG batches."""

import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from torch_geometric.data import Batch

from cdvae.common import data_utils
from cdvae.pl_data.dataset import CrystDataset


@pytest.mark.parametrize("data_config", ["carbon", "mp/mp20new"])
def test_legacy_condition_configs_do_not_require_pdm_dimension(data_config):
    config_dir = Path(data_utils.__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(str(config_dir), version_base="1.1"):
        cfg = compose(config_name="default", overrides=[f"data={data_config}"])
    resolved = OmegaConf.to_container(cfg.model.conditions, resolve=True)
    assert resolved["types"]["element_values"]["n_in"] is None
    embedding = hydra.utils.instantiate(cfg.model.conditions, _recursive_=False)
    assert "composition" in embedding.cond_keys
    assert "element_values" not in embedding.cond_keys


def test_apollox_requires_and_propagates_explicit_pdm_dimension():
    config_dir = Path(data_utils.__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(str(config_dir), version_base="1.1"):
        cfg = compose(config_name="default", overrides=["data=apollox/mp_apollox"])
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.data.pdm_dim
        cfg = compose(
            config_name="default", overrides=["data=apollox/mp_apollox", "data.pdm_dim=3"]
        )
    assert list(cfg.data.prop) == ["pressure", "element_values"]
    assert list(cfg.data.conditions) == ["composition", "element_values"]
    assert list(cfg.data.num_targets) == [1, 3]
    assert cfg.data.datamodule.datasets.train.pdm_dim == 3
    assert cfg.data.datamodule.datasets.val[0].pdm_dim == 3
    assert cfg.data.datamodule.datasets.test[0].pdm_dim == 3
    assert cfg.model.conditions.types.element_values.n_in == 3


@pytest.fixture
def graph_arrays():
    return (
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        np.array([26, 28]),
        np.array([3.0, 3.0, 3.0]),
        np.array([90.0, 90.0, 90.0]),
        np.array([[0, 1], [1, 0]]),
        np.zeros((2, 3), dtype=int),
        2,
    )


@pytest.fixture
def serial_preprocess(monkeypatch, graph_arrays):
    # Crystal parsing and graph construction do not interpret PDM properties.
    monkeypatch.setattr(data_utils, "build_crystal", lambda cif, **kwargs: cif)
    monkeypatch.setattr(data_utils, "build_crystal_graph", lambda *args: graph_arrays)
    monkeypatch.setattr(
        data_utils, "p_umap", lambda function, *args, **kwargs: list(map(function, *args))
    )
    monkeypatch.setattr(hydra.utils, "log", logging.getLogger(__name__), raising=False)


def make_dataset(path, **kwargs):
    options = dict(
        name="PDM test",
        path=str(path),
        force_process=False,
        prop=["pressure", "element_values"],
        pdm_dim=3,
        niggli=False,
        primitive=False,
        graph_method="crystalnn",
        preprocess_workers=1,
        lattice_scale_method="scale_length",
    )
    options.update(kwargs)
    return CrystDataset(**options)


@pytest.mark.parametrize("value", ["[1, 2.5, 3]", [1, 2.5, 3], np.array([1, 2.5, 3])])
def test_parse_and_validate_json_and_native_vectors(value):
    assert data_utils.validate_pdm_vector(value, expected_dim=3) == [1, 2.5, 3]


@pytest.mark.parametrize("value", [2, 2.5, np.float64(2.5), "2.5"])
def test_scalar_properties_still_parse(value):
    assert data_utils.parse_prop(value) == float(value)


@pytest.mark.parametrize(
    "value",
    [[], 1.0, [[1, 2]], [1, float("nan")], [float("inf")], ["1", "2"], [True], "invalid"],
)
def test_reject_invalid_pdm_values(value):
    with pytest.raises(ValueError, match="finite, non-empty 1D numeric vector"):
        data_utils.validate_pdm_vector(value)


def test_reject_wrong_pdm_dimension():
    with pytest.raises(ValueError, match="2 features; expected 3"):
        data_utils.validate_pdm_vector([1, 2], expected_dim=3)


@pytest.mark.parametrize("native", [False, True])
def test_feather_pdm_reaches_scaled_pyg_batch(tmp_path, serial_preprocess, native):
    source = tmp_path / "train.feather"
    values = [[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]]
    pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "cif": ["cif a", "cif b"],
            "pressure": [0.0, 2.0],
            "element_values": values if native else [str(value) for value in values],
        }
    ).to_feather(source)

    dataset = make_dataset(source)
    dataset.prop_scalers = [
        data_utils.get_scaler_from_data_list(dataset.cached_data, key)
        for key in dataset.prop
    ]
    assert dataset[0].element_values.shape == (1, 3)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    assert batch.element_values.shape == (2, 3)
    assert batch.pressure.shape == (2, 1)
    means = torch.tensor([2.0, 4.0, 6.0])
    stds = torch.tensor([1.0, 2.0, 3.0]) + data_utils.EPSILON
    torch.testing.assert_close(batch.element_values, (torch.tensor(values) - means) / stds)

    # An unchanged, valid cache preserves the PDM data and shape on reload.
    reloaded = make_dataset(source)
    assert reloaded.cached_data[0]["element_values"] == values[0]


def test_preprocess_reports_missing_source_property(tmp_path, serial_preprocess):
    source = tmp_path / "train.feather"
    pd.DataFrame({"material_id": ["a"], "cif": ["cif a"], "pressure": [0.0]}).to_feather(source)
    with pytest.raises(ValueError, match="missing configured properties.*element_values"):
        make_dataset(source)
    assert not source.with_suffix(".pkl").exists()


def test_old_cache_requires_explicit_reprocessing(tmp_path, serial_preprocess, graph_arrays):
    source = tmp_path / "train.feather"
    pd.DataFrame(
        {
            "material_id": ["a"],
            "cif": ["cif a"],
            "pressure": [0.0],
            "element_values": [[1.0, 2.0, 3.0]],
        }
    ).to_feather(source)
    old_records = [{"mp_id": "a", "pressure": 0.0, "graph_arrays": graph_arrays}]
    with source.with_suffix(".pkl").open("wb") as cache_file:
        pickle.dump(old_records, cache_file)
    with pytest.raises(ValueError, match="element_values.*force_process=true"):
        make_dataset(source)
    rebuilt = make_dataset(source, force_process=True)
    assert rebuilt.cached_data[0]["element_values"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("pdm_dim", [3, None])
def test_dataset_rejects_mismatched_cached_vectors(tmp_path, serial_preprocess, graph_arrays, pdm_dim):
    source = tmp_path / "train.feather"
    records = [
        {"mp_id": "a", "pressure": 0.0, "element_values": [1, 2, 3], "graph_arrays": graph_arrays},
        {"mp_id": "b", "pressure": 0.0, "element_values": [1, 2], "graph_arrays": graph_arrays},
    ]
    with source.with_suffix(".pkl").open("wb") as cache_file:
        pickle.dump(records, cache_file)
    with pytest.raises(ValueError, match="record 1:.*2 features; expected 3"):
        make_dataset(source, pdm_dim=pdm_dim)


def test_invalid_feather_pdm_does_not_write_cache(tmp_path, serial_preprocess):
    source = tmp_path / "train.feather"
    pd.DataFrame(
        {
            "material_id": ["a"],
            "cif": ["cif a"],
            "pressure": [0.0],
            "element_values": ["[1, NaN, 3]"],
        }
    ).to_feather(source)
    with pytest.raises(ValueError, match="Invalid PDM.*record 0:.*finite"):
        make_dataset(source)
    assert not source.with_suffix(".pkl").exists()


def test_scalar_only_dataset_remains_supported(tmp_path, serial_preprocess, graph_arrays):
    source = tmp_path / "train.feather"
    records = [{"mp_id": "a", "pressure": 4.0, "graph_arrays": graph_arrays}]
    with source.with_suffix(".pkl").open("wb") as cache_file:
        pickle.dump(records, cache_file)
    dataset = make_dataset(source, prop=["pressure"], pdm_dim=None)
    dataset.prop_scalers = [data_utils.StandardScalerTorch(torch.tensor(2.0), torch.tensor(2.0))]
    assert "element_values" not in dataset[0]
    torch.testing.assert_close(dataset[0].pressure, torch.tensor([[1.0]]))
