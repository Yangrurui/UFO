# Adam SP 29-DoF: full G1-aligned training

## Selected profile: domain randomization, fresh training (2026-10-10)

Edit `humanoidverse/config/domain_rand/adam_sp.yaml` for Adam randomization.
It lists enabled effects, ranges, units and sampling timing in one place.
The robot YAML selects `domain_rand=adam_sp` instead of overriding individual
numbers in Hydra strings. `ctrl_delay_step_range` directly controls every motor
group; their duplicated min/max fields have been removed from the new profile.
Motor gains, torque-speed curves and action scales remain in the robot YAML.
Inference delay uses a normal MJLab action class, and mass/inertia use a normal
startup event. No runtime method replacement or extra config loader is needed.

`configs/robots/adam_sp_29dof_ufo_mechanical_dr.yaml` and
`configs/data/adam_sp_g1_full_ufo_mechanical_dr.yaml` provide an opt-in successor
to the mechanical mapping profile. `../train_adam_sp_mechanical_dr.sh` selects
them and a separate `runs/ufo_fb_adam_sp_mechanical_dr_8gpu` directory.
`../train_adam_sp.sh` now selects this launcher. Eight-GPU training started
at 2026-10-10 15:09:28 UTC with random model weights, new optimizers, empty replay
and zero counters; the budget is 192M global samples.

The profile preserves all 29 action scales, explicit XLab PD gains, 862 motion
clips, observation noise, COM ranges, and grouped 0–20 ms actuator delay.
It adds:

| Setting | Effective behavior |
| --- | --- |
| Mass and inertia | One startup uniform multiplier in [0.95, 1.05] per environment/body scales both nominal mass and principal inertia. COM is preserved; repeated application does not compound. |
| Contact friction | Startup sliding coefficient [0.5, 1.25], shared across robot geoms within each environment. The plane's sliding coefficient is 0, so MuJoCo max mixing no longer masks samples below 1. Other contact parameters retain their defaults. |
| Reference reset | Uniform q ±0.05 rad, root xyz ±[0.025, 0.025, 0.005] m, world roll/pitch/yaw ±[0.05, 0.05, 0.1] rad, linear velocity ±[0.25, 0.25, 0.1] m/s and angular velocity ±[0.26, 0.26, 0.39] rad/s. Joint positions respect hard limits; qdot is preserved. |
| Inference delay | User-selected 10–20 ms, sampled as 10/15/20 ms for each new policy command and environment. All joints share that sample. Commands are released on 5 ms physics boundaries, with the 20 ms policy cadence unchanged. |

Inference delay precedes the existing actuator delay, giving nominal combined
command delays of 10–40 ms. This is a training coverage assumption, not a measured
end-to-end hardware latency distribution. PD feedback is current; raw policy
actions and their observation history retain the original semantics. Reset
queues hold the actual reset joint pose, not the first new policy command.

Reference reset perturbations are skipped for explicit target-state resets and
evaluation mode, so the normal tracking evaluation starts exactly at the
reference. Dynamics randomization and inference delay remain enabled in that
evaluation unless `disable_domain_randomization` is selected. That switch also
disables all new effects. Legacy profiles and G1 keep the new options disabled.

Code: `humanoidverse/agents/envs/adam_randomization.py` and
`humanoidverse/agents/envs/inference_delay.py`.
CPU validation: `tests/test_adam_randomization*.py`, `tests/test_inference_delay.py`
and `../adam_sp/randomization_20261010/validate_cpu.py` (actual MJLab CPU smoke).
Validation passed: 97 CPU tests after configuration cleanup, plus a two-environment MJLab simulation with
16 control steps, partial reset, finite state/reward and intact raw action
history. Actual friction draws were 0.88357 and 0.57645, and each body's mass
and inertia used the same scale. Results are recorded in
`../adam_sp/randomization_20261010/validation.json` and `cpu_smoke.json`.
Configuration cleanup preserved all effective randomization, motor, PD and
action-mapping parameters (`config_cleanup_equivalence.json` in that directory).
The previous mechanical run stopped after its complete 83,279,872-step checkpoint.
Its model and optimizer remain available for comparison. All old Adam replay
directories were removed as requested, releasing 306.36 GiB; the new training
imports no model, optimizer or replay state. Launch and cleanup records are
`fresh_start_status.json` and `replay_deletion_result.json` in the same directory.
Startup was verified at 15:14 UTC: evaluation began at step 0, optimizers had
no initialized state, and the first ten gradient synchronizations completed
across all eight GPUs with finite values (`fresh_start_validation.json`).

## Previous profile: mechanical-limit coverage (2026-10-10)

The selected entry point `../train_adam_sp.sh` now delegates to
`../train_adam_sp_mechanical.sh`. It uses
`configs/robots/adam_sp_29dof_ufo_mechanical.yaml`, the corresponding data
manifest, and the separate `runs/ufo_fb_adam_sp_mechanical_8gpu` directory.
The user explicitly selected hip pitch scale **0.55 on both sides** and asked
for the same mechanical-coverage rule on every other joint.

The mapping remains `qPD = q0 + randomized_offset + 5*s*u`. For all other
joints, `s=max(abs(mechanical_low-q0),abs(mechanical_high-q0))/(5*0.9)`.
Scales do not depend on reference extrema or the previous scale. All mirrored
left/right joint pairs now have equal scales. This covers mechanical endpoints;
it does not clamp PD targets to mechanical or soft position limits.

| Joint (both sides where applicable) | Selected s |
| --- | ---: |
| Hip pitch | 0.550000000 |
| Hip roll | 0.356666667 |
| Hip yaw | 0.214444444 |
| Knee | 0.384666667 |
| Ankle pitch | 0.164444444 |
| Ankle roll | 0.077577778 |
| Waist roll | 0.062000000 |
| Waist pitch | 0.302444444 |
| Waist yaw | 0.184222222 |
| Shoulder pitch | 0.802888889 |
| Shoulder roll | 0.598444444 |
| Shoulder yaw | 0.574000000 |
| Elbow | 0.488000000 |
| Wrist yaw | 0.593333333 |
| Wrist pitch / roll | 0.213333333 |

The YAML retains full formula precision. CPU validation compiles the actual
MJCF, checks every mechanical endpoint with default offsets +/-0.02 rad,
and checks the actual adapter's effective scales. All 29 intervals remain
reachable with `|u|<1`: worst case is waist roll at 0.964516; hip pitch uses
0.903273 nominally and 0.910545 with offset. Thus the hip override provides
approximately, rather than exactly, 10% nominal action headroom.
See `../adam_sp/mechanical_mapping_20261010/coverage.json` (relative to UFO).

Training migration waits for the old run's next complete checkpoint at
60,858,368 global steps. The switch preserves its checkpoint and old profile,
copies model and optimizer state to the new run, and imports **no old replay**.
An empty-replay checkpoint recovery now collects at least 10,240 fresh local
transitions per rank before optimization; this collection uses the learned
policy because the original random-action seed phase has already elapsed.
The cumulative budget remains 192M global steps, with eight GPUs and the same
validated communication settings. New-run lineage and
`../adam_sp/mechanical_mapping_20261010/switch_status.json` distinguish a
prepared/queued migration from verified new training updates.

Domain randomization is retained: startup link masses x[0.95,1.05] (inertia is
not rescaled), torso COM +/-2 cm per local axis, robot geom sliding friction
[0.5,1.25], per-joint Kp/Kd x[0.9,1.1]; reset default offsets +/-0.02 rad and
grouped delays of 0--20 ms; training reset includes 30% lie-down initialization.
Every 1--3 seconds, horizontal root velocity receives increments +/-0.5 m/s
per axis and angular velocity receives increments +/-0.5 rad/s per axis.
Actor observations have uniform noise: gyro +/-0.2 rad/s before its 0.25 scale,
gravity +/-0.05, joint positions +/-0.01 rad, velocities +/-0.5 rad/s.
Formal tracking evaluation retains DR and observation noise but starts from
the reference first frame, bypassing random initial phases and lie-down reset.

## Previous profile: reference coverage around the default posture (2026-10-09)

On 2026-10-09 the user selected a single scale for both directions of each
joint, centered on its original default posture. Its retained launcher is
`../train_adam_sp_ufo_range.sh`, using
`configs/robots/adam_sp_29dof_ufo_range.yaml` and its matching data manifest.
The original baseline launcher is retained as `../train_adam_sp_xlab.sh`.

The selected mapping is `qPD = q0 + randomized_offset + 5*s*u`, where the actor
outputs `u` in [-1,1] and `s` is the per-joint value in
`training.actuator.action_scales`. Its original global `control.action_scale`
is not the effective scale when this per-joint override is present. There is
no `action_target_ranges` override in this profile, so the symmetric path runs.
The reference-coverage formula is
`s=max(original_s,max(abs(reference-q0))/(5*0.9))`, over all 258,600 frames.
Kp/Kd, armature, physical limits, motor envelopes, delays and pose settings
remain the original XLab configuration.

| Joint(s) | Original s | Selected s |
| --- | ---: | ---: |
| Left hip pitch | 0.191667 | 0.467880 |
| Right hip pitch | 0.191667 | 0.552000 |
| Hip roll | 0.200000 | 0.356667 |
| Hip yaw | 0.200000 | 0.214444 |
| Knee | 0.191667 | 0.384667 |
| Ankle pitch | 0.153846 | 0.164444 |
| Ankle roll | 0.142857 | 0.142857 |
| Elbow | 0.125000 | 0.488000 |

Both elbows have effective raw-action gain `5*s=2.44 rad`, with nominal PD
targets [-2.74,2.14] rad around q0=-0.3. Without the nominal 10% action margin,
the minimum elbow scale would be 0.4392 (effective gain 2.196). These are PD
command ranges, not mechanical joint limits. The 0.9 coverage statement uses
the nominal posture: under the configured +/-0.02 rad pose offset, worst-case
reference action magnitude is about 0.924324 (ankle pitch), and 0.908197 for
elbows. Left/right hip pitch scales differ because their reference extrema
differ; symmetric here means positive/negative directions of each joint.

The saved symmetric model/optimizer state at 3,203,072 steps / 6,080 optimizer
updates matches the selected profile. On 2026-10-09 the user requested removal
of UFO replay data: all `runs/*/checkpoint/buffers/**/buffer.hdf5` files,
including this run's eight replay shards, were deleted. Model weights,
optimizer states, configurations and evaluation outputs were preserved.
These checkpoints no longer support complete restoration of the original
training state. Each affected checkpoint has `REPLAY_REMOVED.json`;
the deletion manifest is
[`ufo_replay_deletion.json`](../../adam_sp/storage_cleanup_20261009/ufo_replay_deletion.json).

The user requested training from scratch on 2026-10-09. The selected launcher
now uses `runs/ufo_fb_adam_sp_ufo_range_fresh_8gpu`, with a randomly initialized
model, new optimizer state and empty replay. No earlier checkpoint is imported.
The full-training budget is 192,000,000 global steps from zero, using all 862
motions, eight GPUs and 1,024
environments per GPU. Standard warmup is 10,240 local steps per rank. Model,
optimizer and rank replay checkpoints are saved every 3.2M global steps and
at normal completion; later launches resume this experiment's own checkpoint.
`training_lineage.json` records initialization and accounting. Earlier model
checkpoints remain available for comparison.

No new training was started when selecting this profile. At the selection
check, about 92.3 GiB was free; reaching 25.6M with full replay checkpoints
needs about 108.3 GiB of additional space, a minimum shortfall of 15.9 GiB
before allowing further working space. The completed asymmetric result remains
a baseline.

Container cleanup on 2026-10-09 resolved this storage shortfall: it released
45.9 GiB and left 138.1 GiB free, with about 29.8 GiB remaining after the
projected checkpoint growth to 25.6M steps. Disposable early test payloads and
regenerable caches were removed; three original Adam replay shards were
compressed losslessly, with every dataset compared byte-for-byte before
replacement. All four formal model/optimizer/replay checkpoints, datasets,
evaluation artifacts and installed environments were retained. This cleanup
did not start training. See
[`storage_cleanup_20261009/summary.json`](../../adam_sp/storage_cleanup_20261009/summary.json)
for the measured space usage and cleanup records.

## Completed experiment: bounded asymmetric PD targets

The run `runs/ufo_fb_adam_sp_asymmetric_8gpu` completed 25,600,000 steps and
49,824 optimizer updates on 2026-10-08 at 22:42:45 UTC, with a complete final
checkpoint. Its final independent 862-clip evaluation has mean angle L2 error
1.310016 rad. See `../adam_sp/training_review_20261009/review.html` for exact-step
comparisons and four policy videos. It was launched on
2026-10-08 at 10:57:28 UTC through `../train_adam_sp_asymmetric.sh`.
It uses `configs/robots/adam_sp_29dof_ufo_asymmetric.yaml` and
`configs/data/adam_sp_g1_full_ufo_asymmetric.yaml`, with the same 862 clips,
eight GPUs, 1,024 environments per GPU and an initial 25,600,000-step budget.
The earlier symmetric-range experiment was stopped after preserving its first
complete checkpoint and evaluation at 3,203,072 steps. At that point its angle
L2 error was 3.8783 rad, versus original Adam 3.3212 and G1 3.2094; this was an
early comparison, not a conclusion about eventual convergence.

Adam's nominal Kp/Kd are explicit XLab values, independent of its separately
specified armature. They are not recomputed from armature or an effort/Kp
heuristic. The new mapping also retains motor curves, physical limits, delays,
gain randomization and default posture. Only the conversion from policy action
to desired joint position changes.

`training.actuator.action_target_ranges` defines one [L, U] pair per joint:
`L=max(physical_low,min(reference_min,default)-0.05)` and
`U=min(physical_high,max(reference_max,default)+0.05)` in radians, using all
258,600 reference frames. For normalized actor action `u` in [-1, 1], the target
is `center + u*(U-center)` for positive action and
`center + u*(center-L)` for negative action. `center` is the original default
plus its randomized offset, clamped to [L, U]. The result is converted to the
existing MJLab action term's input; its original scale is retained as a unit
conversion. Replay stores actor actions in [-1, 1]; last-action/history retain
the existing native [-5, 5] representation.

For both elbows, the nominal targets for actions -1, 0, +1 are
-2.496, -0.300, +0.209 rad. The prior symmetric expansion permitted +2.140 rad
on the extension side. The new target stays within its configured bounds,
including when the randomized default offset is enabled. This is a commanded
target bound, not a guarantee that physical joint positions cannot overshoot.
Reference samples at physical endpoints have no outward command headroom:
26/29 joints touch at least one physical limit, and 20 touch both. Static
coverage is complete within 1e-6 rad export tolerance, but does not establish
dynamic tracking quality.

Validation includes 42 passing mapping/export/robot/actuator tests, real
32-environment MJLab command checks and two-GPU training/save/resume to
5,120 steps / 36 updates. In the same fixed test action stream, commanded
targets outside physical limits were 11.80% for the original map, 26.57% for
the symmetric expansion and 0% for the asymmetric map. Formal randomization
also gave 0%, with finite state and torque. This is a plumbing/stress check,
not an evaluation of a trained policy. Reports are in
`../adam_sp/tracking_diagnosis_20261008/asymmetric_runtime_validation.json`,
`asymmetric_action_reachability.json` and `xlab_pd_gain_provenance.json`.

The actor ONNX continues to output raw policy actions. Tracking export adds
the new mapping to its `.meta.json`; consumers must apply it before PD control.
The bare actor files from goal/reward export likewise need this external
mapping. Old checkpoints and replay buffers are not directly interchangeable
with the new action semantics, so this experiment starts fresh.

Run `python ../adam_sp/tracking_diagnosis_20261008/first_stage_comparison.py --mapping asymmetric`
to compare complete, identical-step evaluations against original Adam, G1
and the symmetric candidate. The log is
`logs/train_adam_sp_asymmetric_8gpu.log`; launch and checkpoint verification
records are under `../adam_sp/asymmetric_run_20261008/`.

## 2026-10-08: tracking diagnosis and final checkpoint fix

The original XLab-profile run finished normally at 191,995,904 global steps.
The old loop did not save its final partial checkpoint interval, so only the
188,981,248-step state survives. The missing 3,014,656 steps of updates cannot
be recovered from logs. `Workspace.train_online()` now saves every fully
completed rollout/update at normal budget termination if the periodic
checkpoint has not already saved that state. Failed updates do not overwrite
the previous checkpoint through this final-save path. This addresses normal
termination; it does not provide crash-atomic or SIGKILL-time saving.

Fifteen loop regression tests cover rounded and exact budgets, resume,
distributed step accounting, rank buffer files, duplicate-save avoidance, and
failed/nonfinite updates. A real two-GPU Adam test with formal randomization
saved at 1,024 steps for a budget of 1,027, restored model/optimizer/replay and
saved at 2,048 for a budget of 2,053, then performed no rewrite when rerun with
an already-completed 2,048-step budget. See
`runs/adam_sp_final_save_validation/validation_*.json` and
`../adam_sp/validate_checkpoint_resume.py`.

The tracking audit found an action-interface mismatch: XLab's Adam PPO uses
an unbounded Gaussian mean, while UFO uses a tanh actor in [-1, 1], scaled to
native actions in [-5, 5]. With the original XLab scales, 14.36% of Adam's
joint/frame reference angles lie outside the nominal PD target command range,
versus 1.55% for G1 over the identical 862 clips. Both Adam elbows exceed that
range in about 60% of dataset frames. This is a command coverage measure,
not a bound on actual physical motion or tracking error. The runtime checks
and full per-joint audit are in
`../adam_sp/tracking_diagnosis_20261008/action_reachability.json`.

A separate candidate, `configs/robots/adam_sp_29dof_ufo_range.yaml`, changes
only action scales and descriptive metadata. Each scale is the larger of the
original scale and `max(abs(reference - default)) / (5 * 0.9)`, evaluated on
all 258,600 frames. Twenty-six scales increase; the reference set then fits
within 90% of the actor range. Robot geometry, motor envelopes, PD gains,
delays, randomization, default pose, dataset, and FB algorithm stay the same.
This does not establish dynamic trackability or convergence. The candidate
passed a small two-GPU gradient/save/resume test: 4,096 steps / 30 updates,
then 5,120 steps / 36 updates, with zero parameter/buffer difference between
ranks at both saves. Model/optimizer and both replay shards were restored.
It is not a newly trained production policy. Its manifest is
`configs/data/adam_sp_g1_full_ufo_range.yaml`.

Four deterministic-action diagnostic rollouts (motion IDs 609, 373, 578, 103)
measure average joint-angle L2 distance at 1.53163 rad for Adam and 0.99429
for G1. Increasing only the old Adam policy's elbow action scales by two
changes its mean distance to 1.52743. Fight improves 1.98676 → 1.86310,
while walk worsens 0.97351 → 1.22536. Fight elbow MAE improves
0.64067 → 0.32728 rad, consistent with the command coverage diagnosis,
but the mixed overall result demonstrates that this post-hoc scale change
is not a replacement for retraining. These four training clips and one seed
do not estimate full-dataset improvement. See
`../adam_sp/tracking_diagnosis_20261008/diagnostics_summary.json` and
`tracking_diagnosis.png` in the same directory. Evaluation disables observation
noise; physical randomization, actuator delay and random default-pose offsets
remain enabled, matching the official evaluation path.

Use a fresh run for this mapping: directly resuming the old policy/replay with
different action scales changes the meaning of recorded actions and actor
outputs. The original configuration, formal run and model-only review copy
are preserved. The prepared first-stage launcher is:

```bash
./train_adam_sp_ufo_range.sh
```

It uses the same eight-GPU/1,024-env-per-rank setup and starts with a
25,600,000-step budget in `runs/ufo_fb_adam_sp_ufo_range_8gpu`. This fresh run
was launched on 2026-10-08 at 09:16:52 UTC in detached screen session
`ufo_fb_adam_sp_ufo_range_8gpu`. Output is in
`logs/train_adam_sp_ufo_range_8gpu.log`; launch details and the eventual exit
code are recorded under `../adam_sp/ufo_range_run_20261008/`. Compare its complete 862-clip evaluation
curve with the original Adam curve at equal steps before extending the budget.
`../adam_sp/tracking_diagnosis_20261008/first_stage_comparison.py` reads the
three runs and writes exact-step comparisons only after all 862 unique motion
records are present. The separate one-shot `check_first_checkpoint.py` under
the launch-record directory waits for the first periodic checkpoint and
records its eight replay shards, model/optimizer file completeness and rank
synchronization in `first_checkpoint_verification.json`.
Measure per-joint saturation and the fight/dance/get-up categories as well as
the aggregate. Prioritize action coverage before changing reward weights or
removing XLab's hardware randomization. The original run documentation below
describes the preserved baseline.

The current training manifest is `configs/data/adam_sp_g1_full.yaml`.
The full-data preparation and validation result is in
`../adam_sp/full_validation.json`; the launcher requires that record to pass.
The [current reference review](../../adam_sp/comparison_full/review.html)
contains the G1 quality comparison, per-clip results and nine motion videos.
The [earlier audit](ADAM_SP_AUDIT.md) describes problems in the superseded data.

## What matches G1

The Adam training set uses exactly the same 40 LaFAN1 source sequences and
862 non-overlapping, 300-frame windows as the saved G1 baseline: 258,600 frames at
30 Hz, including all 90 fall/get-up clips. Motion keys, key order and original
source intervals match G1. Partial tails are omitted in both datasets; no
leading frames or difficult clips are removed. All subjects, including subject
5, are used for training, matching G1. Evaluation on this manifest measures
training-set performance; it is not a held-out subject test.

`../adam_sp/g1_dataset_selection.json` records the selection. The original
77 LaFAN1 BVHs are in `../adam_sp/source_data/bvh`; only the selected 40 are
retargeted. The original dataset license remains in
`../adam_sp/source_data/LAFAN1_license.txt`.

The FB architecture, optimization settings, rewards, termination conditions,
observation noise, history, timing, and 30% lie-down
initialization match G1. Adam has 64 actor observations, 372 history values,
463 privileged observations (including root height), and 29 actions. The Hydra
`max_local_self` dimension is declared as 462; the runtime adds root height,
as in G1. These dimensions are checked against the actual saved checkpoints. Its virtual head is placed at the
actual Adam head collision center: `[0.03, 0, 0.47]` relative to the torso.
`../adam_sp/comparison_full/config_comparison.json` records the resolved
configuration comparison. Five domain-randomization fields intentionally differ:
Adam enables grouped command delays of 0–4 physics steps and PD-gain scaling
of 0.9–1.1, following XLab. G1 disables those two randomizations. Other resolved
non-robot environment settings match.

## Adam-specific model and retargeting

Robot config: `configs/robots/adam_sp_29dof.yaml`.
Hydra config: `humanoidverse/config/robot/adam_sp/adam_sp_29dof.yaml`.
MJCF: `humanoidverse/data/robots/adam_sp/adam_sp_29dof.xml`.
Model provenance: `../adam_sp/robot_source.json`.

The model and 29-DoF motor profile come from the local XLab Adam SP resources.
Link geometry, mass and inertia, joint order and axes, limits, default pose,
armature, friction, torque/velocity limits and PD gains remain Adam-specific.
The hand endpoints are the wrist-roll links. Fixed finger geometry is fused
in the MJCF so MuJoCo and UFO Torch FK use the same 30-body skeleton.
The virtual head makes the augmented body count 31, matching G1's observation
structure without copying G1's body dimensions.

The selected XLab asset is `adam_sp/29dof_torque_speed_envelope_delay4_v1`.
Its twelve leg joints use six four-quadrant torque–speed envelope groups;
the remaining joints use four native MuJoCo PD groups (position and velocity
channels). The port preserves measured motoring/braking curves, peak limits,
and current-velocity torque clipping. Each group samples one independent lag
per environment at reset: 0, 5, 10, 15 or 20 ms, held within the episode. Position,
velocity and feedforward commands are delayed; feedback remains current.
Native PD channels remain grouped because joint-by-joint sorting interleaves
position and velocity channels incorrectly. Runtime tests cover this mapping.
Hip roll/yaw action scales are 0.20. Other action scales retain XLab's nominal
values rather than being recomputed from the new peak torque ceilings.

The exported source profile is `../adam_sp/xlab_adam_training_profile.json`.
Source files were read from `/root/XLab/source/xlab/xlab/assets/adam_sp/mjlab.py`,
`actuators/{mjlab,torque_speed_envelope,compiler}.py` and the Adam whole-body
shadowing task configurations. The port in `humanoidverse/agents/envs/actuators/`
retains XLab's license and attribution. XLab's PPO/MoE, motion sampling and
task-specific reward settings are not copied into the UFO FB experiment.
UFO's soft position-limit reward factor remains 0.95 to match G1; XLab's asset
uses 0.90 for its own soft-limit setting. Neither changes Adam's physical limits.

Retargeting uses GMR commit `bb1bbe40774794fceb2a7c579a3464a28e68c844`
with Mink/DAQP in `../adam_sp/.venv`. `retarget_v2.py` retains the original
frames and timing, with Adam-specific shoulder anchors, limb proportions and
forearm coordinates. The solver uses all-body ground constraints, native
self-collision geometry, penetration recovery, physical joint limits and 98%
of a conservative reference velocity budget. For leg joints this budget is
the smaller of the previous nominal limit and 95% of the measured zero-torque
motoring speed. Physical motor metadata retain XLab's actual values. This
prevents requesting speeds above the powered envelope; it is not a proof of
required torque or dynamic balance. QP limits are expressed in configuration
increments. Both ends of a capsule and all corners of a box are constrained,
including when an elbow lies horizontally on the floor. A nonlinear contact
check accepts each IK step. Temporal root objectives reduce jitter without
slowing the source motion. There is no after-the-fact root-height shift.

Every frame is replayed in an independent MuJoCo model with a floor. The audit
checks finite values, unit quaternions, joint limits, joint speed including
original clip boundaries, collision penetration, FK and solver failures.
A sequence stuck at an identical pose for more than 10 consecutive transitions
also stops the export. The replay thresholds are 3 mm floor penetration,
10 mm self-penetration, 1e-5 rad joint-limit error and 1.05 speed/limit ratio;
the report preserves actual values and soft metrics such as root acceleration,
stance foot speed, clearance and target errors. These gates are local checks,
not an official G1 acceptance standard. Collision comparisons use each
robot's native geometry, so geometry alone affects those results.

Current raw retargets: `../adam_sp/retargeted_g1_xlab/`.
Training PKL: `cache/motion_data/adam_sp_g1_full_xlab/adam_sp_g1_full_10s_ufo.pkl`.
Full-sequence PKL: `cache/motion_data/adam_sp_g1_full_xlab/adam_sp_g1_full_full_ufo.pkl`.
The custom export preserves G1's exact clip names and order. The old 121-clip
training and 60-clip subject-5 manifests are retained as historical artifacts;
they are not selected by the formal launcher.

## Validation and launch

The completion record includes independent replay of all 258,600 frames,
31 passing configuration/export/actuator tests, native-body and virtual-head FK
checks, and a 4,096-step / 60-update FB gradient test with formal randomization.
The final checkpoint also passed reload and a 64-step tracking inference run.
The gradient test uses 16 environments and a 128-wide network to fit alongside
G1. It validates the training interface, not long-run convergence. Its final
checkpoint is `runs/adam_sp_xlab_smoke/checkpoint`; its log is
`logs/adam_sp_xlab_smoke.log`. A separate 32-environment runtime check verifies
all ten actuator groups, observed delays, gain ranges and finite torques.

Full-frame replay reports zero floor/self penetration and zero failed QPs.
Stance foot speed P95 is 0.464 m/s (G1 0.327); after dividing by leg length,
the values are 0.627 versus 0.639 per second. Root acceleration P95 is
15.33 m/s² (G1 7.88); after the same normalization Adam is still about 34%
higher. Stance clearance P95 is 5.23 mm (G1 14.86). These are kinematic quality
metrics, not policy success rates. Nine reference videos have sampled-frame
visual review, including four fall/get-up windows.

From the workspace root, when the selected GPUs are available:

```bash
./train_adam_sp.sh
```

This runs FB on 8 GPUs, 1,024 environments per GPU, 192,000,000 total environment
steps, buffer size 5,120,000, a checkpoint interval of 3,200,000 global steps,
latent refresh interval 100, and seed 4728, matching G1. The run directory is
`runs/ufo_fb_adam_sp_xlab_8gpu`. A validated full dataset is required before
launch. An existing checkpoint in that run directory is resumed automatically.
Do not resume the old Adam checkpoints without the head observation using the
new observation configuration. G1 training uses its own run directory and data.

On 2026-10-01 the existing G1 process group was stopped to release the eight
GPUs for this Adam run. Its complete checkpoint at 76,873,728 global steps
remains in `runs/ufo_fb_g1_8gpu/checkpoint`. The switch record is
`../adam_sp/training_switch.json`; Adam startup evidence is in
`../adam_sp/formal_training_status.json`, with live output in
`logs/train_adam_sp_xlab_8gpu.log`. The detached screen session is named
`ufo_fb_adam_sp_xlab_8gpu`.

To reproduce full preparation and the final small training check, from the
workspace root (GPU 1 needs enough free memory for the small test):

```bash
python adam_sp/rebuild_full_dataset.py
```

This rebuilds all 40 sequences, audits and exports the data, resolves the G1
configuration comparison, renders reference videos, runs the gradient test,
and writes the validation record. Logs are saved in `adam_sp/` and `UFO/logs/`.
For a fresh gradient test, use a new work directory or preserve and move the
previous smoke run first, because UFO automatically resumes existing runs.

From UFO, inspect a checkpoint against the full dataset:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=2 \
  .venv/bin/python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_adam_sp_xlab_8gpu \
  --robot-config configs/robots/adam_sp_29dof.yaml \
  --data-manifest configs/data/adam_sp_g1_full.yaml --dataset adam_sp_g1_full \
  --device cuda:0 --headless --disable-dr --disable-obs-noise \
  --save-mp4 --motion-list 0 --max-steps 300
```

Reference videos show kinematic Adam motion beside the original human skeleton.
They do not show the learned policy. Some retargeted poses differ from the
human because Adam's body and motors differ; use the numerical comparison and
videos together. Dynamic tracking success still needs evaluation after
training. Hardware deployment is outside this simulation adaptation.
