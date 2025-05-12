# Actions

An action is composed of several tasks, each of which can have multiple runs.
Actions are serial in tasks.
Tasks are parallel in runs.
Actions have a single codebase that is not mutated during runs.

Actions are user-submitted asks that can be performed on the data (not equivalent to jobs). Includes:

[Runners] Single-task, single-run (equivalent to jobs):
- train and test an ML model
- retest a trained model
- analyze test results

[Tasks] Single-task, multi-run:
- hyperparameter search (parallel jobs)
- multi-model scan (parallel jobs)
- other matrix jobs
- analysis / plot tasks (over several models)

[Chains] Multi-task, multi-run:
- dependent job chains...

Several actions can be initiated at once from the `action_parameters` file (in parallel for now).