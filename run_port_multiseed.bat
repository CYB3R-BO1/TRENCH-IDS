@echo off
setlocal

cd /d "%~dp0"
set LOG=run_port_multiseed.log
set PYTHONUNBUFFERED=1

echo ============================================== > "%LOG%"
echo Starting port multi-seed comparison run >> "%LOG%"
echo %date% %time% >> "%LOG%"
echo ============================================== >> "%LOG%"

echo [1/4] train seed=42 newport >> "%LOG%"
echo %time% >> "%LOG%"
.venv\Scripts\python.exe -m trench_ids.cl.train replay.enabled=true replay.buffer_size_per_task=200 replay.replay_fraction=0.3 ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.seed=42 paths.out_dir=runs/replay_seed42_newport >> "%LOG%" 2>&1
if errorlevel 1 (
    echo TRAIN seed=42 FAILED, aborting. >> "%LOG%"
    goto :done
)

echo [2/4] evaluate seed=42 newport >> "%LOG%"
echo %time% >> "%LOG%"
.venv\Scripts\python.exe -m trench_ids.cl.evaluate --run-dir runs/replay_seed42_newport >> "%LOG%" 2>&1
if errorlevel 1 (
    echo EVALUATE seed=42 FAILED, aborting. >> "%LOG%"
    goto :done
)

echo [3/4] train seed=1 newport >> "%LOG%"
echo %time% >> "%LOG%"
.venv\Scripts\python.exe -m trench_ids.cl.train replay.enabled=true replay.buffer_size_per_task=200 replay.replay_fraction=0.3 ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.seed=1 paths.out_dir=runs/replay_seed1_newport >> "%LOG%" 2>&1
if errorlevel 1 (
    echo TRAIN seed=1 FAILED, aborting. >> "%LOG%"
    goto :done
)

echo [4/4] evaluate seed=1 newport >> "%LOG%"
echo %time% >> "%LOG%"
.venv\Scripts\python.exe -m trench_ids.cl.evaluate --run-dir runs/replay_seed1_newport >> "%LOG%" 2>&1
if errorlevel 1 (
    echo EVALUATE seed=1 FAILED, aborting. >> "%LOG%"
    goto :done
)

echo ============================================== >> "%LOG%"
echo ALL DONE >> "%LOG%"
echo %date% %time% >> "%LOG%"
echo ============================================== >> "%LOG%"

:done
echo.
echo Finished (or stopped early on error). See %LOG% for full output.
