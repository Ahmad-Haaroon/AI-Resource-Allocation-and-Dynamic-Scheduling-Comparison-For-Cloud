# Stage 2: learned Decima compatibility and pilot

## Scope

Use the original simulator, GNN architecture, policy loss, gradient computation,
input-dependent baseline and training curriculum. This is a TensorFlow 2.15
compatibility port, not an exact TensorFlow 1.x numerical reproduction. No claim
of trained-policy performance is supported by a three-update pilot.

The original checkout remains unchanged. `prepare_decima_compat.py` generates an
ignored copy under `external/`, checks the upstream commit and records changed
source hashes. Our repository contains the adapter and preparation instructions,
not the upstream code/data.

## Compatibility changes

- Import `tensorflow.compat.v1` and disable TensorFlow 2 behavior.
- Replace the eight `contrib.layers.fully_connected` calls through a minimal
  adapter preserving layer widths, activation functions, Glorot-uniform weight
  initialization, zero biases, and `weights`/`biases` variable names.
- Replace the removed `np.bool` alias with `bool`.
- Use public `tf.convert_to_tensor` instead of the removed private ops symbol.
- Launch upstream training with process cleanup and a 900-second pilot timeout;
  failed runs retain their logs and exit status. This does not implement exact
  training resumption: upstream does not save curriculum/average-reward history.

Reference: https://www.tensorflow.org/guide/migrate/upgrade documents compat.v1
migration and notes that removed contrib APIs need additional changes.

## Environment and commands

Intel macOS and Python 3.9 were used. This environment is separate from Stage 1,
which used NumPy 2.0.2; TensorFlow 2.15 requires an older NumPy range.

```sh
python3 -m venv .venv-decima
.venv-decima/bin/python -m pip install -r requirements-stage2.txt
.venv-decima/bin/python -B scripts/prepare_decima_compat.py \
  --source '../Reference Project/decima-sim' --destination external/decima-compat
.venv-decima/bin/python -B scripts/check_decima.py \
  --source external/decima-compat --output results/stage2/graph_check_v2
.venv-decima/bin/python -B scripts/run_training_pilot.py \
  --source external/decima-compat --output results/stage2/pilot
.venv-decima/bin/python -B scripts/check_decima.py \
  --source external/decima-compat --output results/stage2/trained_check \
  --checkpoint results/stage2/pilot/models/model_ep_3
```

Output directories must not exist. Select new names for repetitions. Preparation
also refuses to overwrite an existing compatibility copy. For an independent
checkout, use the upstream acquisition instructions in README.md.

## Pilot design

`configs/training-pilot.json` predeclares 50 executors, one initial and 200 streaming
jobs, 25 s mean inter-arrival time, two rollout workers, and three attempted
updates (`num_ep=4`, since upstream uses an exclusive loop bound). It retains the
README's stochastic termination and differential reward settings. Workers are
reduced from 16 to 2 for the local pilot, and each TensorFlow process uses one
intra-op and one inter-op thread. Random episode termination can mean that only a
subset of the 201 generated jobs finish during a training rollout.

Final evaluation seeds, training budgets and learned-versus-heuristic performance
comparisons are later-stage work. Seed 9000 is used only for compatibility checks
and must not become a final held-out seed.

## Environment setup issue observed

The first install attempt selected google-auth 2.50.0, which pulled cryptography
50.x as a source distribution. Its Rust setup failed in an unwritable user cache.
That attempt was canceled. The requirements now pin google-auth 2.29.0 and
its OAuth helper to 1.2.0, and the successful setup must use prebuilt wheels:

```sh
.venv-decima/bin/python -m pip install --only-binary=:all: \
  --cache-dir external/pip-cache -r requirements-stage2.txt
```

This is an environment failure, not an experimental result. The initial console
error is summarized here; `results/stage2/install.log` records the retry.

## Corrected earlier manuscript claim

The original graph check rejected the report's unverified 12,736-parameter claim.
The actual released default architecture has 4,770 trainable parameters:
GCN 1,008 + graph summaries 784 + stage head 1,633 + executor head 1,345.
The initial failure remains in `results/stage2/graph_check.log`;
`graph_check_v2` records successful inference and exact checkpoint round-tripping.
The test now checks the count independently against these layer dimensions.
The final manuscript must use this measured count for this implementation.

Dependency validation (`pip check`) passed. The interpreter's LibreSSL produces
an urllib3 compatibility warning on TensorFlow import. No network calls are used
in the simulator or training pipeline; this is not a claim that HTTPS clients in
this interpreter are supported. TensorFlow v1 API deprecation warnings are expected.
