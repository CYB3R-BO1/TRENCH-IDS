import importlib


def test_stage_b_meta_training_module_imports():
    mod = importlib.import_module("trench_ids.cl.train_meta")
    assert hasattr(mod, "main")
