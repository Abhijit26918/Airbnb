# ATTRIBUTION POLICY — HIGHEST PRIORITY, NO EXCEPTIONS

Abhijit Kumar is the ONLY author and contributor of this repository.

NEVER, under any circumstance, in any commit message, PR title, PR body,
issue, code comment, docstring, README, or any other file:
  - add a `Co-Authored-By:` trailer of any kind
  - add "Generated with Claude Code" or any variant
  - add a 🤖 emoji footer or attribution block
  - mention Claude, Anthropic, an AI assistant, or a bot as a contributor

Commit messages are plain Conventional Commits and end at the body.
There is no trailer section. If you are unsure whether to add a line at the
bottom of a commit message: do not add it.

Before running `git commit`, re-read this block.

## Project rules

- Python 3.11. Package manager: uv. Line length 100. Ruff + ruff-format.
- Type hints on every public function. `from __future__ import annotations`.
- No logic in notebooks. Notebooks import from `src/pricelens/`. Never the reverse.
- No hardcoded paths, no hardcoded city names, no magic numbers in `src/`. Everything
  from `configs/`.
- Every stochastic operation takes an explicit `seed` argument. No bare
  `np.random.*`. Folds are read from `data/processed/folds.parquet`, never regenerated.
- Never write a feature that consumes the target without out-of-fold computation.
- Never add a column from the kill-list in configs/city/*.yaml `drop_columns`.
- Write the test before the feature for anything in `features/text.py` and `data/clean.py`.
- One phase = one PR = one focused set of commits. Conventional Commits.
- After changing anything under `features/` or `models/`, run `make audit` and paste
  the result into the PR body.
- If a change makes CV improve by more than 0.05 R² in one step, STOP and assume a
  leak until proven otherwise. Report it, don't celebrate it.
- Do not `pip install` anything not in pyproject.toml. Add it there first.
- This machine has no `make` binary on PATH. Makefile targets are thin wrappers over
  `uv run pricelens <command>` (typer CLI) — when `make` isn't available, run the
  underlying `uv run pricelens ...` / `uv run pytest` / `uv run ruff` command directly.
  Keep the Makefile correct regardless; it's exercised by CI and by anyone on Mac/Linux.

## Commit message format
    <type>(<scope>): <subject>

    <body — what and why, not how>

  Nothing after the body. No trailers. No footers. See the attribution policy above.
