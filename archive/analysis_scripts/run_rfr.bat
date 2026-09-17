@echo off
conda activate base
python -m trench_ids.cl.train train rfr.enabled=true rfr.lambda=0.1 rfr.apply_only_base=true replay.enabled=true train.epochs_per_task=1 train.warmup_epochs=1 train.batch_size=32 paths.out_dir=runs/rfr_test