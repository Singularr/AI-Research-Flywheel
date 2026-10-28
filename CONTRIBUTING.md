# Contributing

The goal of this project is to improve AI research performance through controlled, reproducible experiments—not through unsupported claims.

## Workflow

1. Open an issue or describe the problem, hypothesis, and intended outcome.
2. Record the current baseline before changing behavior.
3. Make a small, focused change.
4. Add or update tests for the intended behavior.
5. Run the relevant tests and evaluation against the same baseline conditions.
6. Report results, including regressions, failed runs, and limitations.
7. Submit a pull request for review. Changes to core experiment, evaluation, scoring, or decision logic require explicit human approval.

## Evidence requirements

For performance-related changes, include:
- The hypothesis being tested.
- Baseline and candidate measurements, with metric definitions.
- The evaluation procedure, dataset/task set, and relevant configuration.
- Reproducibility details, including a seed where applicable.
- Failures, uncertainty, and known limitations.

Do not describe a change as an improvement unless the measurements support that conclusion. Do not remove failed experiment results simply because they are unfavorable.

## Code and tests

- Keep changes focused and avoid unrelated refactors.
- Add or update tests for behavior changes.
- Run the relevant tests before requesting review and include the commands and results in the PR.
- Do not add dependencies without explaining why they are needed.
- Never commit secrets, credentials, private data, or local environment files.
- Update documentation when usage, configuration, or evaluation behavior changes.

## Pull requests

- Explain the problem, the proposed change, and how it was validated.
- Call out changes to core logic explicitly.
- Include measured results for claims about performance.
- Mark tests or evaluations that were not run; do not imply they passed.
