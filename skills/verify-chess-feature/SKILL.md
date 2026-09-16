---
name: verify-chess-feature
description: Generate Python verification code for chess SAE/LoRSA/transcoder features from backend feature.interpretation fields. Use when Codex needs to turn an interpreted chess feature, taxonomy label, rationale, or Mongo feature record into executable validation code using this repo's src/feature_varification evaluator and rules.
---

# Verify Chess Feature

Use this skill to write repo-native Python code that tests whether a chess SAE feature matches its stored interpretation.

## Workflow

1. Read the feature identity: `sae_name`, `sae_series`, `feature_index`, layer, and feature type.
2. Read `feature.interpretation`. Prefer structured fields:
   - `taxonomy`
   - `text`, usually prefixed like `[Mov]`
   - `rationale`
   - `source`
3. Infer the verification target from both the taxonomy and rationale. Do not rely on the label alone.
4. Generate Python that uses `src.feature_varification`, especially:
   - `FeatureSpec`, `VerificationCase`, `ThresholdSpec`
   - `evaluate_feature_rule`
   - rule classes from `src.feature_varification.rules`
5. Keep generated code explicit about assumptions, required model/SAE loading objects, dataset inputs, and thresholds.

## Interpretation Parsing

Use this fallback order for taxonomy:

```python
taxonomy = interpretation.get("taxonomy")
if not taxonomy:
    text = interpretation.get("text", "")
    taxonomy = text.split("]", 1)[0].lstrip("[") if text.startswith("[") else None
```

Use `rationale` plus the non-prefix part of `text` as the semantic description.

When a feature has only `[Label]` and no rationale, generate a scaffold with TODOs for the rule choice instead of inventing a precise test.

## Rule Selection

Map common interpretation language to existing rules:

- Occupancy / "contains Own rook" / "black rook occupancy": `PieceTypeRule("own r")` or `PieceTypeRule("opponent r")`.
- Source square / origin of top move: registered rule `"move_start_square"`; requires `move_uci`.
- Target square / destination of top move: registered rule `"move_end_square"`; requires `move_uci`.
- King area / around king: `"own_king_neighborhood"` or `"opponent_king_neighborhood"`.
- Queen checking around opponent king: `"queen_check_around_opponent_king"`.
- Bishop/queen diagonal movement field: `PieceRayRule(piece_type, directions=("diagonal",), stop_at_blockers=...)` or `PieceDestinationRule(piece_type)`.
- Rook/queen rank/file movement field: `PieceRayRule(piece_type, directions=("rank", "file"), ...)` or `PieceDestinationRule(piece_type)`.
- Knight movement field: `PieceDestinationRule("own n")` or `PieceDestinationRule("opponent n")`.
- Pawn forward pattern: `front_file_rule(piece_type)` or `front_cone_rule(piece_type)`.
- Protection relation (`Pro`) or capture relation (`Cap`) may require a custom `FunctionalRule` if no existing rule captures the relation.
- Tactical relation (`Tac`) often needs an `AnyOfRule`/`AllOfRule` composition or a custom callable; write the callable locally and wrap it in `FunctionalRule`.
- Register (`Reg`) and spatial (`Spa`) are often weak semantics; generate diagnostic code comparing against a broad spatial rule and leave the target rule easy to edit.
- Value (`Val`) is not usually a square-mask feature. Generate code that compares activation aggregates against WDL/value metadata rather than `evaluate_feature_rule`, unless the rationale names a spatial rule.

Piece selectors use project convention: `"own p"`, `"own n"`, `"own b"`, `"own r"`, `"own q"`, `"own k"`, and the corresponding `"opponent ..."` forms.

## Code Shape

For square-mask features, generate a script with this shape:

```python
from src.feature_varification import FeatureSpec, ThresholdSpec, VerificationCase, evaluate_feature_rule
from src.feature_varification.rules import PieceTypeRule, PieceDestinationRule, PieceRayRule

feature = FeatureSpec(feature_type="lorsa", layer=0, feature_id=8262)
cases = [
    VerificationCase(fen="...", move_uci="...", label="sample_0"),
]
rule = PieceDestinationRule("opponent b")
threshold = ThresholdSpec(mode="ratio_to_max", value=0.7, scope="sample")

result = evaluate_feature_rule(
    model=model,
    lorsas=lorsas,
    transcoders=transcoders,
    feature=feature,
    rule=rule,
    cases=cases,
    threshold=threshold,
)
print(result.to_dict())
```

If model/SAE loading is not obvious in the local context, leave `model`, `lorsas`, and `transcoders` as explicit parameters or TODO variables rather than guessing paths.

## Output Standards

- Include the original interpretation as a comment at the top.
- Include why the selected rule matches the rationale.
- Include at least one threshold choice; default to `ThresholdSpec(mode="ratio_to_max", value=0.7, scope="sample")` for top-square style features.
- If the rule requires moves, validate that every `VerificationCase` has `move_uci`.
- Save results as JSON when producing a standalone script.
- Prefer adding a small script under `scripts/feature_verification/` for reusable validations.

For exact available rule constructors and evaluator arguments, inspect `src/feature_varification/rules.py` and `src/feature_varification/evaluator.py` in the target repo before finalizing code.
