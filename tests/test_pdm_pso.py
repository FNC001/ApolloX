"""Regression coverage for the PDM CSV -> PSO generation command boundary."""

import importlib.util
from pathlib import Path
import shlex

import pandas as pd
import pytest
from apollox_common.pdm import build_schema, save_schema


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_sh = load_module("pdm_make_sh", "PSO/make_sh.py")
standardize = load_module("pdm_standardize", "prepare_dataset/standardize.py")


def prepare_generation_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "temp").mkdir()
    poscars = tmp_path / "poscars" / "generation1"
    poscars.mkdir(parents=True)
    (poscars / "POSCAR-test").write_text(
        "example\n1\n10 0 0\n0 10 0\n0 0 10\nB Co Cr Fe Mo Ni\n1 1 1 1 1 1\n"
    )


def generated_element_values(tmp_path):
    script = (tmp_path / "temp/sh_files_1/run_evaluation_0.sh").read_text()
    argument = next(part for part in shlex.split(script)
                    if part.startswith("--element_values="))
    return [float(value) for value in argument.split("=", 1)[1].split(",")]


def test_standardize_to_generated_command_preserves_all_21_pdm_features(tmp_path, monkeypatch):
    prepare_generation_directory(tmp_path, monkeypatch)
    elements = ["B", "Co", "Cr", "Fe", "Mo", "Ni"]
    columns = sorted(a + b for i, a in enumerate(elements) for b in elements[i:])
    means = list(range(21))
    raw_values = [4 * (i + 1) + means[i] for i in range(21)]
    data = pd.DataFrame([["particle-0", "particle-0.cif", *raw_values, -3.2]],
                        columns=["material_id", "cif_file", *columns, "Energy"])
    source = tmp_path / "optimized.csv"
    data.to_csv(source, index=False)
    stats = tmp_path / "scaler_stats.txt"
    stats.write_text("Train Mean: " + ",".join(map(str, means)) + "\n"
                     "Train Std: " + ",".join(["2"] * 21) + "\n")
    save_schema(tmp_path / "pdm_schema.json", build_schema(elements, ["pair"], 5.0))
    output = tmp_path / "temp/standardized_optimized_particles_distribution1.csv"

    standardize.main(source, stats, output)
    make_sh.generate_scripts(1, apollox_path=str(ROOT))

    assert list(pd.read_csv(output).columns) == ["material_id", "cif_file", *columns]
    values = generated_element_values(tmp_path)
    assert len(values) == 21
    assert values == [float(2 * (i + 1)) for i in range(21)]


def test_make_sh_excludes_energy_by_name_without_truncating_pdm(tmp_path, monkeypatch):
    prepare_generation_directory(tmp_path, monkeypatch)
    data = pd.DataFrame([["particle-0", "particle-0.cif", 0.5, -5.0, 1.5, 2.5]],
                        columns=["material_id", "cif_file", "CoCo", "Energy", "CoNi", "NiNi"])
    data.to_csv(tmp_path / "temp/standardized_optimized_particles_distribution1.csv", index=False)

    make_sh.generate_scripts(1, apollox_path=str(ROOT))

    assert generated_element_values(tmp_path) == [0.5, 1.5, 2.5]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), "missing"])
def test_make_sh_rejects_invalid_pdm_instead_of_shortening_vector(tmp_path, monkeypatch, invalid_value):
    prepare_generation_directory(tmp_path, monkeypatch)
    data = pd.DataFrame([["particle-0", "particle-0.cif", 0.5, invalid_value, 2.5]],
                        columns=["material_id", "cif_file", "CoCo", "CoNi", "NiNi"])
    data.to_csv(tmp_path / "temp/standardized_optimized_particles_distribution1.csv", index=False)

    with pytest.raises(ValueError):
        make_sh.generate_scripts(1, apollox_path=str(ROOT))

    assert not (tmp_path / "temp/sh_files_1/run_evaluation_0.sh").exists()


@pytest.mark.parametrize("stats_text", [
    "Train Mean: 1,2\nTrain Std: 1\n",
    "Train Mean: 1,2\nTrain Std: 1,0\n",
    "Train Mean: 1,nan\nTrain Std: 1,1\n",
])
def test_standardize_rejects_inconsistent_scaler_statistics(tmp_path, stats_text):
    stats = tmp_path / "scaler_stats.txt"
    stats.write_text(stats_text)
    with pytest.raises(ValueError):
        standardize.load_scaler_stats(stats)


def test_standardize_reports_feature_count_without_metadata(tmp_path):
    source = tmp_path / "pdm.csv"
    pd.DataFrame([["sample", "CoNi", 1.0, 2.0]],
                 columns=["label", "formula", "CoCo", "CoNi"]).to_csv(source, index=False)
    stats = tmp_path / "scaler_stats.txt"
    stats.write_text("Train Mean: 1\nTrain Std: 1\n")

    with pytest.raises(ValueError, match="2 PDM features, but 1 scaler values"):
        standardize.main(source, stats, tmp_path / "scaled.csv", allow_legacy_order=True)
