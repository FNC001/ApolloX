"""Exercise real subprocess failures and batch output retention without GPU inference."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_batch", ROOT / "use_model/submit_tasks.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.mark.parametrize("value", [True, False, 0, -1, "1#comment", 1.5, None])
def test_bad_batch_count_fails_before_jobs(value):
    with pytest.raises(ValueError, match="positive integer"):
        batch.positive_integer(value, "num_batches_to_samples")


def test_gpu_assignment_preserves_parent_physical_ids():
    assert batch.gpu_devices({"CUDA_VISIBLE_DEVICES": "2,5"}) == ["2", "5"]
    assert batch.gpu_devices({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert batch.gpu_devices({"CUDA_VISIBLE_DEVICES": "-1"}) == []


def test_legacy_requires_opt_in_and_does_not_mix_with_v2(tmp_path):
    model, source = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    source.mkdir()
    with pytest.raises(ValueError, match="matching pdm_schema"):
        batch.matching_schema(model, source)
    assert batch.matching_schema(model, source, True) is None
    (source / "pdm_schema.json").write_text("{}")
    with pytest.raises(ValueError, match="matching pdm_schema"):
        batch.matching_schema(model, source, True)


def fake_scripts(tmp_path, *, generation_exit=0, extraction_exit=0, exported_count=1):
    root = tmp_path / "fake repo"
    scripts = root / "cond-cdvae/scripts"
    scripts.mkdir(parents=True)
    (scripts / "evaluate.py").write_text(
        "import argparse, os\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--model_path');p.add_argument('--label')\n"
        "a,_=p.parse_known_args();d=Path(a.model_path)\n"
        "(d/('eval_gen_'+a.label+'.pt')).write_text('recoverable result')\n"
        "(d/'child-device.txt').write_text(str(os.environ.get('CUDA_VISIBLE_DEVICES')))\n"
        f"raise SystemExit({generation_exit})\n"
    )
    (scripts / "extract_gen.py").write_text(
        "import sys,csv\nfrom pathlib import Path\n"
        f"if {extraction_exit}: raise SystemExit({extraction_exit})\n"
        "p=Path(sys.argv[1]);d=p.with_suffix('')/'gen';d.mkdir(parents=True)\n"
        "rows=[]\n"
        f"for i in range({exported_count}):\n"
        " f=d/(str(i)+'.vasp');f.write_text('H\\n1\\n5 0 0\\n0 5 0\\n0 0 5\\nH\\n1\\nDirect\\n0 0 0\\n')\n"
        " rows.append(dict(filename=f.name,material_id='generated',eval_index=0,sample_index=i))\n"
        "with (d/'structure_manifest.csv').open('w') as f:\n"
        " w=csv.DictWriter(f,fieldnames=['filename','material_id','eval_index','sample_index']);w.writeheader();w.writerows(rows)\n"
    )
    model, logs, output = tmp_path / "model", tmp_path / "logs", tmp_path / "output"
    for path in (model, logs, output):
        path.mkdir()
    return root, model, logs, output


def execute_job(paths, *, keep=True, expected_count=1):
    root, model, logs, output = paths
    return batch.generate_job("unique-job", "POSCAR.test.vasp", "H1", [1.0], model,
                              logs, output, expected_count, 1, 1, device="5",
                              keep_intermediate=keep, root=root)


def test_extension_label_and_device_survive_real_subprocesses(tmp_path):
    paths = fake_scripts(tmp_path)
    rows = execute_job(paths)
    root, model, logs, output = paths
    assert rows[0]["material_id"] == "POSCAR.test.vasp"
    assert (output / rows[0]["filename"]).is_file()
    assert (model / "child-device.txt").read_text() == "5"
    assert (model / "eval_gen_unique-job.pt").is_file()


@pytest.mark.parametrize("failed_stage", ["generation", "extraction"])
def test_failed_stage_never_deletes_original_tensor(tmp_path, failed_stage):
    paths = fake_scripts(tmp_path, generation_exit=2 if failed_stage == "generation" else 0,
                         extraction_exit=2 if failed_stage == "extraction" else 0)
    with pytest.raises(subprocess.CalledProcessError):
        execute_job(paths, keep=False)
    assert (paths[1] / "eval_gen_unique-job.pt").read_text() == "recoverable result"
    assert list(paths[3].iterdir()) == []


def test_incomplete_manifest_retains_tensor_and_publishes_no_structures(tmp_path):
    paths = fake_scripts(tmp_path, exported_count=1)
    with pytest.raises(ValueError, match="Incomplete extraction"):
        execute_job(paths, keep=False, expected_count=2)
    assert (paths[1] / "eval_gen_unique-job.pt").is_file()
    assert list(paths[3].iterdir()) == []


def test_explicit_cleanup_only_after_complete_export(tmp_path):
    paths = fake_scripts(tmp_path)
    rows = execute_job(paths, keep=False)
    assert (paths[3] / rows[0]["filename"]).is_file()
    assert not (paths[1] / "eval_gen_unique-job.pt").exists()
    manifest = pd.read_csv(paths[3] / "manifests/unique-job.csv")
    assert manifest.material_id.tolist() == ["POSCAR.test.vasp"]
    assert manifest.filename.tolist() == [rows[0]["filename"]]


def test_failed_batch_returns_failure_and_saves_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "prepare_inputs", lambda *a: ([("sample", "H1", [1])], None))
    monkeypatch.setattr(batch, "gpu_devices", lambda: [])
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")
    monkeypatch.setattr(batch, "generate_job", fail)
    with pytest.raises(RuntimeError, match="1 generation jobs failed"):
        batch.run_batch({"batch_size": 1, "num_batches_to_samples": 1}, tmp_path)
    summary = json.loads(next((tmp_path / "parallel_logs").glob("*/summary.json")).read_text())
    assert len(summary["failures"]) == 1
    assert summary["structures"] == 0


def test_different_training_scaler_is_rejected(tmp_path):
    model, source = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    source.mkdir()
    (model / "scaler_stats.txt").write_text("Train Mean: 1,2\nTrain Std: 1,1\n")
    scaler = source / "scaler_stats.txt"
    scaler.write_text("Train Mean: 2,3\nTrain Std: 1,1\n")
    with pytest.raises(ValueError, match="External scaler differs"):
        batch.check_external_scaler(model, scaler)


def test_raw_csv_labels_preserve_leading_zeros_and_na(tmp_path):
    from apollox_common.pdm import build_schema, save_schema
    model, source, work = [tmp_path / name for name in ("model", "source", "work")]
    for directory in (model, source, work):
        directory.mkdir()
    schema = build_schema(["H"], ["pair"], 3.0)
    for directory in (model, source):
        save_schema(directory / "pdm_schema.json", schema)
        (directory / "scaler_stats.txt").write_text("Train Mean: 0\nTrain Std: 1\n")
    csv_path = source / "raw.csv"
    pd.DataFrame({"label": ["001", "1", "NA"], "formula": ["H1"] * 3,
                  schema["features"][0]: [1, 2, 3]}).to_csv(csv_path, index=False)
    config = {"mode_choice": "unscaled_pdm", "unscaled_pdm_mode_params": {
        "input_csv": str(csv_path), "scaler_path": str(source / "scaler_stats.txt")}}
    records, actual_schema = batch.prepare_inputs(config, model, work)
    assert [row[0] for row in records] == ["001", "1", "NA"]
    assert actual_schema == schema
