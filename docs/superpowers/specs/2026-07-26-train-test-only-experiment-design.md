# Train/Test-Only Experiment Design

## Goal

The experiment workflow uses only training and test datasets. Development datasets are not configured, loaded, evaluated, fingerprinted, or represented in progress and output artifacts.

## Configuration

`RunConfig` exposes `train_paths` and `test_paths` only. All shipped experiment configurations remove `dev_paths`. Because configuration models are strict, an old configuration containing `dev_paths` fails validation and directs the user to update it rather than silently ignoring unused data.

## Runtime flow

`ExperimentRunner.run` accepts `(train, test)`. It processes training samples in deterministic mini-batches, reveals labels after each successful batch prediction, and updates the experience store exactly as before. Checkpoints remain training boundaries and are recorded in the manifest, but they no longer trigger evaluation. After all training batches finish, the runner evaluates the test set once without learning from it.

The CLI fingerprints and loads only configured train and test files. Progress events and prediction rows use only the `train` and `test` split names. No `metrics-dev-*` artifacts are generated.

## Outputs

The final test metrics remain in `metrics-test-final.json` and `metrics.json`. Training predictions, test predictions, costs, manifests, experience data, and generalized-experience artifacts retain their current formats except that no dev rows or metrics exist.

## Testing

Tests first establish that a configuration without `dev_paths` loads successfully and that a configuration containing `dev_paths` is rejected. Runner tests call `run(train, test)` and assert that checkpoint processing does not produce dev evaluations or dev progress events. CLI/config tests verify all shipped configurations use only train and test paths. The complete offline integration workflow must pass without external APIs or credentials.

## Compatibility

This is an intentional configuration and Python API breaking change. Users must delete `dev_paths` from custom YAML files and update direct runner calls from `run(train, dev, test)` to `run(train, test)`.
