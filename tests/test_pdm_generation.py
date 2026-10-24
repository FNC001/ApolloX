"""Exercise CLI/CSV PDM validation and scaling without a trained decoder."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch

from cdvae.common.condition_utils import pdm_tensor_for_model, prepare_pdm_values
from cdvae.common.data_utils import StandardScalerTorch


class ConditionRecorder:
    """Record inputs at the embedding boundary; replace expensive structure decoding."""

    def __init__(self, pdm):
        self.cond_keys = ["composition"] + (["element_values"] if pdm else [])
        self.sub_emb_list = [SimpleNamespace(n_out=1)]
        if pdm:
            self.sub_emb_list.append(SimpleNamespace(n_in=3))
        self.seen = []

    def __call__(self, conditions):
        self.seen.append(conditions)
        batch_size = len(conditions["composition"][1])
        result = {"composition": torch.ones(batch_size, 1)}
        if "element_values" in self.cond_keys:
            result["element_values"] = conditions["element_values"]
        return result


class DecoderDouble:
    def __init__(self, pdm=True):
        self.device = torch.device("cpu")
        self.multiemb = ConditionRecorder(pdm)
        self.hparams = SimpleNamespace(
            latent_dim=2,
            data=SimpleNamespace(prop=["pressure"] + (["element_values"] if pdm else [])),
        )
        self.prop_scalers = [StandardScalerTorch(torch.tensor([0.0]), torch.tensor([2.0]))]
        if pdm:
            self.prop_scalers.append(StandardScalerTorch(
                torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 5.0])
            ))
        self.lattice_scaler = None
        self.latents = []

    def zgivenc(self, z, conditions):
        return torch.cat([z, *conditions.values()], dim=-1)

    def langevin_dynamics(self, z, kwargs, num_atoms, atom_types):
        self.latents.append(z)
        return {
            "frac_coords": torch.zeros(int(num_atoms.sum()), 3),
            "num_atoms": num_atoms,
            "atom_types": atom_types,
            "lengths": torch.ones(len(num_atoms), 3),
            "angles": torch.full((len(num_atoms), 3), 90.0),
        }


@pytest.fixture
def evaluation(monkeypatch):
    # Only bypass eval_utils' checkpoint-loading dependency tree. The evaluated
    # script, PDM helpers, tensors and scaler implementations are the real code.
    stub = ModuleType("eval_utils")
    stub.load_model = lambda *args, **kwargs: None
    stub.composition2atom_types = lambda comp: [
        element.Z for element, count in comp.items() for _ in range(int(count))
    ]
    monkeypatch.setitem(sys.modules, "eval_utils", stub)
    path = Path(__file__).resolve().parents[1] / "cond-cdvae/scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("pdm_evaluate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_args(tmp_path, **overrides):
    options = dict(
        model_path=str(tmp_path), tasks=["gen"], start_from="data",
        pressure=0, spgno=1, element_values="3, 10, 18", n_step_each=1,
        step_lr=1e-4, min_sigma=0, save_traj=False, disable_bar=True,
        num_batches_to_samples=1, num_evals=1, batch_size=2,
        down_sample_traj_step=1, formula="H2", hydride=False,
        train_data=None, energy_per_atom=-1, energy=-1,
        enthalpy_per_atom=-1, enthalpy=-1, label="",
    )
    options.update(overrides)
    return SimpleNamespace(**options)


def test_cli_scaler_matches_training_transform_and_keeps_property_order():
    model = DecoderDouble()
    expected = model.prop_scalers[1].transform(torch.tensor([[3.0, 10.0, 18.0]]))
    actual = prepare_pdm_values(
        model, "3,10,18", ["element_values", "pressure"], list(reversed(model.prop_scalers))
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize("value", ["1,2", "1,nan,3", "1,inf,3", "1,no,3", "", "[[1,2,3]]"])
def test_invalid_cli_pdm_fails_before_generation(value):
    model = DecoderDouble()
    with pytest.raises(ValueError, match="element_values"):
        prepare_pdm_values(model, value, model.hparams.data.prop, model.prop_scalers)


def test_missing_or_unused_pdm_is_explicit():
    with pytest.raises(ValueError, match="requires element_values"):
        pdm_tensor_for_model(DecoderDouble(), None)
    composition_only = DecoderDouble(pdm=False)
    assert pdm_tensor_for_model(composition_only, None) is None
    with pytest.raises(ValueError, match="does not enable PDM"):
        pdm_tensor_for_model(composition_only, "1,2,3")


@pytest.mark.parametrize("failure", ["absent", "count", "width", "shape", "zero_std", "nan_mean"])
def test_missing_or_incompatible_saved_scaler_fails(failure):
    model = DecoderDouble()
    names = model.hparams.data.prop
    scalers = model.prop_scalers
    if failure == "absent":
        names = ["pressure", "energy"]
    elif failure == "count":
        scalers = scalers[:1]
    elif failure == "width":
        scalers[1].means = torch.zeros(2)
    elif failure == "shape":
        scalers[1].means = torch.zeros(3, 1)
    elif failure == "zero_std":
        scalers[1].stds[0] = 0
    elif failure == "nan_mean":
        scalers[1].means[0] = float("nan")
    with pytest.raises(ValueError, match="scaler"):
        prepare_pdm_values(model, "1,2,3", names, scalers)


def test_main_generation_scales_pdm_once_and_keeps_scalar_conditions(evaluation, tmp_path, monkeypatch):
    model = DecoderDouble()
    cfg = SimpleNamespace(data=model.hparams.data)
    monkeypatch.setattr(evaluation, "load_model", lambda *a, **kw: (model, None, cfg))
    monkeypatch.setattr(evaluation.torch, "save", lambda *a, **kw: None)
    evaluation.main(cli_args(tmp_path, pressure=4))
    seen = model.multiemb.seen[0]
    torch.testing.assert_close(seen["element_values"], torch.tensor([[1., 2., 3.], [1., 2., 3.]]))
    torch.testing.assert_close(seen["pressure"], torch.full((2, 1), 2.0))
    torch.testing.assert_close(seen["energy"], torch.full((2, 1), -1.0))


def test_composition_only_generation_without_pdm(evaluation, tmp_path, monkeypatch):
    model = DecoderDouble(pdm=False)
    monkeypatch.setattr(evaluation, "load_model", lambda *a, **kw: (model, None, SimpleNamespace(data=model.hparams.data)))
    monkeypatch.setattr(evaluation.torch, "save", lambda *a, **kw: None)
    evaluation.main(cli_args(tmp_path, element_values=None))
    assert "element_values" not in model.multiemb.seen[0]


def test_reconstruction_does_not_require_cli_pdm(evaluation, tmp_path, monkeypatch):
    model = DecoderDouble()
    monkeypatch.setattr(evaluation, "load_model", lambda *a, **kw: (model, [], SimpleNamespace(data=model.hparams.data)))
    monkeypatch.setattr(evaluation, "reconstruction", lambda *a, **kw: (None,) * 8)
    monkeypatch.setattr(evaluation.torch, "save", lambda *a, **kw: None)
    evaluation.main(cli_args(tmp_path, tasks=["recon"], element_values=None))


def test_direct_generation_checks_active_pdm_condition(evaluation):
    kwargs = dict(ld_kwargs=SimpleNamespace(save_traj=False), num_batches_to_sample=1,
                  num_samples_per_z=1, batch_size=2, formula="H2")
    with pytest.raises(ValueError, match="does not enable PDM"):
        evaluation.generation(DecoderDouble(pdm=False), element_values=[1, 2, 3], **kwargs)
    with pytest.raises(ValueError, match="requires element_values"):
        evaluation.generation(DecoderDouble(), **kwargs)


def test_target_csv_scales_vectors_without_flattening_samples(evaluation, tmp_path):
    target = tmp_path / "target.csv"
    pd.DataFrame({
        "formula": ["H2", "H2", "H2"], "pressure": [0, 0, 0],
        "element_values": ["[3,10,18]", "5,18,28", "[7,26,38]"],
    }).to_csv(target, index=False)
    model = DecoderDouble()
    evaluation.target_generation(model, SimpleNamespace(save_traj=False), 1, 1, str(target), batch_size=2)
    torch.testing.assert_close(model.multiemb.seen[0]["element_values"], torch.tensor([[1., 2., 3.], [2., 4., 5.]]))
    torch.testing.assert_close(model.multiemb.seen[1]["element_values"], torch.tensor([[3., 6., 7.]]))


def test_target_csv_invalid_pdm_row_is_not_silently_skipped(evaluation, tmp_path):
    target = tmp_path / "target.csv"
    pd.DataFrame({"formula": ["H2", "H2"], "pressure": [0, 0],
                  "element_values": ["[1,2,3]", "[1,2]"]}).to_csv(target, index=False)
    with pytest.raises(ValueError, match="batch row 1"):
        evaluation.target_generation(DecoderDouble(), SimpleNamespace(save_traj=False), 1, 1, str(target))
