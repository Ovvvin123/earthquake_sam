V7 local WSL runtime (no formal training started by the migration)

Code and runtime dataset:
  /home/ovvvin/projects/earthquake-sam
Python environment (existing):
  /home/ovvvin/.venvs/earthquake-sam/bin/python
Profile:
  experiments/california_v7_30m_1024_wsl.json
Live output:
  /home/ovvvin/earthquake-sam-results/runs/v7_30m_1024_lora_lr2_ema250_350_seed7_r01
Windows backup:
  F:\Study————————\UCLA\Earthquake-SAM_交接\training_backups\v7_30m_1024_lora_lr2_ema250_350_seed7_r01

Use WSL and cd to the code directory. Check only (does not start training):
  /home/ovvvin/.venvs/earthquake-sam/bin/python run_experiment.py --sequence experiments/california_v7_30m_1024_wsl.json

Only adding --execute to that command starts the 500-epoch run. Do not start
until 1024px supervised and teacher/UniMatch memory checks have been completed.
An inference-only test does not prove that backward + optimizer + teacher fits.

Settings: 1024px full tiles, D4 only, batch=1, accumulation=8, LoRA rank8,
head base LR=2e-4, LoRA base LR=1e-4, original LR decay retained.
EMA teacher is first created at epoch250. Pseudo-label pixel weight starts
at0.05 and reaches0.20 at350; UniMatch weight reaches0.10 at350; 500 epochs.
The V7 dataset contains valid land DEM input; invalid ocean pixels are excluded.
The legacy marine dataset/loader remains available through its old profiles.

All high-frequency weights, resume checkpoints, metrics, console log and previews
stay under the Linux output path. At completed epochs20,40,... plus249,350,500,
the synchronous backup hook copies the full resume checkpoint and closed small
metadata files to a new epoch directory, checks SHA-256, then publishes it.
Epoch500 also backs up best_student/best_teacher weights when available.
It never copies the live console log or every preview. No automatic cleanup.
Backup failures leave .partial-* directories, emit a warning and are recorded in
the Linux run's backup_events.jsonl; training and the Linux checkpoint survive.
Existing epoch backup directories are never overwritten.

Windows originals and QA gallery remain untouched. No raw DEM/GeoTIFF/PNG trees
are copied to WSL. migration_verification.json lists all copied files and hashes.
After future source edits, synchronize reviewed files and refresh their manifest
before launching; there is no bidirectional auto-sync. Do not edit a running run.
For later reviewed code-only changes, run the Windows source copy_v7_to_wsl.py
inside WSL with --sync-code. It checks the previous destination hashes first,
refuses independent WSL edits, and never replaces the dataset or base weights.
WSL remains on the same computer; the F-drive backup is not an off-machine backup.

Resume from a backup requires the same verified runtime/data paths and an
explicit --emaResume pointing to its ema_checkpoint.pth. The full checkpoint
contains optimizer, RNG, teacher (if activated), config and training state.
Do not feed model.pth to --emaResume or bypass dataset/checkpoint identity checks.

Completed verification:
  63 regression tests: PASS.
  Full production-reader preflight: PASS (174 train,40 validation,22 updates/epoch).
  Split lists: 174 train,40 validation,126 test; no shared keys.
  Real full-tile/legacy fixed-crop equality, eight D4 transforms, preview input: PASS.
  GPU inference: PASS, input1x7x1024x1024, output1x3x1024x1024.
  GPU: RTX4060 Laptop; peak allocated5062531584 bytes (about4.71GiB).
  No backward, optimizer step, full training or teacher-stage memory test ran.
  Code and output paths resolve to /dev/sdd ext4.
Reports remain in WSL:
  /home/ovvvin/earthquake-sam-results/checks/v7_migration_20261030/
  /home/ovvvin/earthquake-sam-results/checks/v7_preflight_20261030/
