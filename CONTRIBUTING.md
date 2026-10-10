# Contributing

1. Set up the project environment with its current ML dependencies; see README.
2. Run the available tests with `python -m unittest discover -s tests`.
3. Keep model shapes, initialization, tokenizer behavior, dataset IDs/configs/splits, training defaults, quantization behavior, and existing paths compatible unless a change is explicitly approved and documented.
4. Use type hints for public helpers, report data provenance, and do not claim a benchmark without a reproducible run.
5. Dataset contributions must include source, config, split, version where available, license evidence, and preprocessing details. Do not commit datasets or checkpoints.
6. Pull requests should explain compatibility impact and include focused tests. Review the applicable code and dataset license terms before redistribution.
