# Codex skills

This directory contains the project-specific Codex skills used to analyze and
verify chess features in Leela-SAEs.

## Included skills

- `annotate-circuit-taxonomy`: drafts reviewable circuit-taxonomy labels.
- `score-chess-autointerp`: scores interpretation consistency and complexity.
- `verify-chess-feature`: generates repository-native feature verification code.

Each skill is self-contained. To install one locally, copy its complete
directory into `~/.codex/skills/`, preserving the directory name and internal
layout. Restart Codex after installation so it can discover the skill.

The board-rendering helper in `annotate-circuit-taxonomy` requires the
`python-chess` package, which is already a dependency of this project.
