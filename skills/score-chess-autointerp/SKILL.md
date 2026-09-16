---
name: score-chess-autointerp
description: Score chess feature autointerpretations for activation consistency and semantic complexity from exported JSON/JSONL top-activation evidence. Use when Codex needs to evaluate LoRSA, Transcoder, or MLP chess features, audit whether an interpretation fits individual activation boards, or produce per-feature 1-5 consistency/complexity scores with reviewable evidence.
---

# Score Chess Autointerp

Assign two independent integer scores to every feature: `activation_consistency` and `complexity`. Inspect every exported top-activation sample; do not score from the interpretation text or aggregate counts alone.

## Workflow

1. Read each JSONL feature row and identify `dictionary_name`, `feature_index`, `feature_type`, layer, interpretation, feature statistics, and every top activation sample.
2. Convert pieces to Own/Opponent using the FEN side to move. FEN ranks run 8 to 1. Treat board index orientation exactly as supplied by the exporter.
3. For each sample, record activated squares and occupants. For LoRSA also record the directional activated-square/high-z-square relation. Check movement reachability, pawn advances/control, protection, captures, lines, tactical coordination, and stable spatial structure.
4. Remove a leading taxonomy prefix such as `[Det]` only when reading the semantic interpretation; do not treat the taxonomy label as evidence that the interpretation is correct.
5. State one concrete unifying hypothesis. Test it against every sample and count clear fits, partial fits, and deviations. A visually plausible example is not a fit unless the named chess relation is present.
6. Score activation consistency from the fit audit. Then score complexity from the minimum chess reasoning required to express the unifying feature. Do not reward vague wording or a long interpretation.
7. Emit one JSON object per unique `(dictionary_name, feature_index)`. If repeated rows disagree, inspect both evidence sets and report the conflict rather than silently averaging.

## Activation Consistency

- `5`: Clear pattern with no deviating examples.
- `4`: Clear pattern with one or two deviating examples.
- `3`: Clear overall pattern, but quite a few examples do not fit.
- `2`: Broad consistent theme but no stable structured relation.
- `1`: No discernible pattern.

Use the full number of exported samples. For the usual 20-sample export, one or two deviations means exactly 1-2; “quite a few” normally means at least 3 while a clear majority still fits. Missing/unreadable samples are not automatic deviations: report them separately and lower confidence. Do not give 5 unless every usable sample supports the same concrete rule.

## Complexity

- `5`: Rich feature firing in diverse contexts with an interesting unifying tactical/strategic theme, for example check-forced queen loss, a promotion race with rook support, or a multi-stage forced sequence.
- `4`: High-level semantic structure requiring multiple interacting chess predicates, for example capturing the Opponent queen with check, multi-piece attacking squares, a pin/recapture/overload, or queen-bishop coordination.
- `3`: Moderate chess structure described by one movement or control relation, for example Own queen coverage, knight reachable squares, pawn advance squares, or a consistent capture/protection relation.
- `2`: One piece/token concept under a stable positional condition, for example the front pawn, a back-rank rook, or the Own queen beside the king.
- `1`: A single piece/token identity without an additional condition, for example Own queen.

Score the simplest faithful explanation. A detector does not become complex merely because its boards contain tactics. Require the feature activation itself to select the richer relation. Conversely, diverse surface pieces can still have high complexity when all samples share one multi-piece or forced-sequence theme.

## Type-Specific Evidence

- LoRSA: use both activations and z-pairs. Identify the pieces on both ends and test the directional relation on each board.
- Transcoder: sparse activations may identify only the output object. Use board context to test whether a second relation consistently selects that object.
- MLP: there is no z-pattern. Rely on activated-square occupancy, geometry, move/control relations, dataset fields, and interpretation fit.

## Output

Write JSONL with these required fields:

```json
{"dictionary_name":"BT4_lorsa_L9A_k30_e16","feature_index":123,"feature_type":"lorsa","layer":9,"interpretation":"Own queen coverage","activation_consistency":4,"complexity":3,"usable_samples":20,"fit_samples":18,"partial_samples":1,"deviating_samples":1,"confidence":0.88,"consistency_rationale":"Concrete sample audit.","complexity_rationale":"Minimum relations needed.","evidence_summary":"Representative fitting and deviating boards/squares."}
```

Use integer scores from 1 through 5. Use `confidence` from 0 to 1. Name representative squares and Own/Opponent pieces in rationales. If the interpretation is empty, derive a working hypothesis from the boards, set `interpretation_source` to `inferred_for_scoring`, and lower confidence; otherwise set it to `existing`.

Before finishing, validate score ranges, unique feature keys, sample-count arithmetic, and that every score has concrete evidence rather than taxonomy-only reasoning.
