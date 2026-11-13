# Contributing

Contributions should keep documentation and implementation aligned. Before proposing a change, inspect the affected modules and update the README or cards when behavior, defaults, supported formats, or limitations change.

## Development setup

Use Python 3.10 or later and install a hardware-appropriate PyTorch build, then install the project in editable mode:

```powershell
python -m pip install -e .
```

Optional CUDA 8-bit optimizer dependencies can be installed with `python -m pip install -e ".[8bit]"`.

## Contribution guidance

- Preserve checkpoint, tokenizer, model-shape, CLI, and data-format compatibility unless the change intentionally updates those interfaces and documents the migration.
- Keep public helpers typed where practical; describe data provenance and licensing for any new dataset integration.
- Do not commit datasets, model checkpoints, credentials, or personal information.
- Do not claim training quality or benchmark results without identifying the checkpoint, evaluation procedure, and reproducible evidence.
- The repository currently has no checked-in `tests/` suite. Do not rely on `python -m unittest discover -s tests` as a project validation command. Include focused verification appropriate to the change in the contribution description; add automated tests when a test suite is introduced.
- Review the repository license and relevant third-party terms before redistributing artifacts.
