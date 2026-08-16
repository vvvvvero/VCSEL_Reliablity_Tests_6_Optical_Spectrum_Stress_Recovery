import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
MOD_PATH = os.path.join(ROOT, "14_stage4b_ar1_guided_generator.py")

spec = importlib.util.spec_from_file_location("stage4b_mod", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_smoke_dataset_has_expected_keys():
    dataset = mod.build_smoke_test_dataset(n_devices=2, seq_len=8)

    assert dataset["x"].shape[0] == 2
    assert dataset["x"].shape[1] == 8
    assert dataset["mask"].shape == (2, 8)
    assert dataset["times_h"].shape == (2, 8)
    assert dataset["T_K"].shape == (2,)
    assert dataset["split"]["train"] == [0]
    assert dataset["split"]["val"] == [1]
    assert dataset["split"]["test"] == [1]
