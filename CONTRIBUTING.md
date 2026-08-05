# Contributing

Keep changes scoped to one pipeline or behavior and include focused validation.

Before opening a pull request:

```bash
python -m compileall starVLA
find scripts examples -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Do not commit:

- model checkpoints or optimizer states;
- datasets, generated labels, or evaluation videos;
- logs, W&B run directories, or simulator outputs;
- machine-specific absolute paths;
- API keys or access tokens.

Use environment variables for local paths and credentials. Keep upstream
StarVLA baseline behavior separate from W²-VLA-specific changes so baseline
comparisons remain reproducible.
