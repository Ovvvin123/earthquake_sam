V5 then V6: explicit profiles, one shared trainer

Both: original SAM-2 initialization, frozen SAM base, Q/V LoRA r8 alpha16,
seed7, d0=.01, BCEWithLogits + .5 FocalTversky, pu=0, EMA decay=.98,
positive pseudo labels + logits consistency, corrected accumulation (2 x 4).
Legacy epoch-based LR schedule is retained. No old weights are resumed.

V5: earth_dataset_v5, 500 epochs, label_known enabled.
Epochs 1..120: supervised + EMA only.
Epochs 121..150: pseudo-label weight .05; consistency ramps from epoch 121.
Epochs 151..240: pseudo-label weight ramps to .2; consistency reaches .1 at 240.
Epochs 241..500: both weights remain at their maxima.
V6: earth_dataset_greece_nz_stride64_v2, 200 epochs, original catalog supervision.
Original pseudo schedule 41 / 50 / 80, consistency 41 / 80.
EMA updates after successful optimizer steps from the beginning, in BOTH runs.
Preview every 5 epochs and full checkpoint every 20 epochs are unchanged.

V5 keeps the original split lists. Only original train/val keys with known
pixels in valid input are retained in memory; source HDF5/lists are read-only.
Minimum known pixels = 1 intentionally retains every tile with any supervision.
Training: 256 crops from a 32-pixel origin grid, sampled in proportion to known
area, followed by synchronized D4. No scale augmentation in this opt-in mode.
Validation: fixed maximum-known-area crop per tile; no augmentation.
V6 retains its original random crop/scale/D4 training and fixed-center validation.
V5 validation still overlaps train geography; these are monitoring metrics.
Unknown V5 pixels can receive positive pseudo labels; label_known does not
prohibit pseudo labels outside the approved coastal envelope.

From PowerShell, in SAM code:
  .\run_v5_then_v6.ps1
This checks both datasets and CLI configurations WITHOUT starting training.
Optional NEW report directory:
  .\run_v5_then_v6.ps1 -ReportDir diagnostics/my_new_preflight

Only when ready to train:
  .\run_v5_then_v6.ps1 -Execute
Keep the launcher running. It starts V5 first, waits for exit 0 AND verifies
history epoch=500 plus final EMA next_epoch/config/identity before starting V6.
V5 failure, interruption, incomplete checkpoint or changed source/data stops
the queue. No automatic retry, resume, or repeated launch is performed.

Equivalent WSL command, from SAM code:
  /home/ovvvin/.venvs/earthquake-sam/bin/python -u run_experiment.py --sequence experiments/california_ds5.json experiments/greece_nz_original.json --execute
Omit --execute for validation only.

New result directories (reserved exclusively only when executing):
  runs/california_ds5_bceft_lora8_ema_unimatch_seed7_r01
  runs/greece_nz_bceft_lora8_ema_unimatch_seed7_r01
Existing directories are refused, including reserved directories after a failed
queue. Investigate before choosing new IDs; never delete prior results to retry.
Each run has console.log, launch.json, resolved_config.json, model_config.json,
checkpoints, teacher/student results and data/source hashes.
