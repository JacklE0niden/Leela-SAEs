---
name: annotate-circuit-taxonomy
description: Draft and review chess feature taxonomy labels for the Circuit Taxonomy Annotation workflow. Use when Codex needs to classify SAE/Transcoder/Lorsa chess features into Det, Src, Tgt, Val, Cap, Pro, Mov, Tac, Reg, Spa, or Uninterpretable from feature interpretations, top activation samples, activation squares, high-z Lorsa z-pattern squares, WDL/value signals, policy moves, target moves, or circuit feature evidence; especially when producing JSON/JSONL proposals for the LLM Review Queue UI before final annotation.
---

# Annotate Circuit Taxonomy

## Core Rule

Draft candidate taxonomy labels, not final annotations, unless the user explicitly asks to write through to the annotation API. Prefer high-confidence labels with short evidence. Use `Uninterpretable` when evidence does not clearly match the taxonomy.

If an input row already has `existing_interpretation`, normally treat it as a hard constraint and do not revise it. The layer prior is the sole exception: an existing `Src` or `Tgt` interpretation in layers 0-7 is invalid and must be reinterpreted as `Det`, `Mov`, `Pro`, or `Tac` from board evidence. For other existing interpretations, do not add fresh chess or edge reasoning to `rationale`; write only `Existing interpretation [X].` (with the actual label). Put no boilerplate edge audit after it.

Use `Src` and `Tgt` only in layers 8-14. Never assign either label in layers 0-7; early move-origin/destination-looking patterns are usually `Det`, `Mov`, `Pro`, or `Tac`. In layers 8-14, consider `Src` and `Tgt` when move evidence is strong, and apply them most aggressively in layers 13-14. `Src`/`Tgt` can appear in three common forms: a single specific piece type on one board, a single activated square on one board with the matching Top Move for that square, or multiple activated squares on one board where every activated square is covered by Top Moves. Do not under-label these later-layer features just because the surface pattern is not the most obvious one.

Read `references/taxonomy-rubric.md` and `references/human-review-lessons.md` before classifying features.

## Workflow

1. Gather structured evidence for each feature:
   - `dictionary_name`, `feature_index`, layer, feature type, circuit file, feature position in circuit.
   - Existing interpretation text.
   - Top activation FENs and the strongest activated squares.
   - For Lorsa features, high-z source squares and directional relations `activated square <- high-z square`.
   - Spend enough reasoning budget to inspect individual top activation samples, not only aggregate counts. For each ambiguous feature, explicitly identify which board squares are activated, which pieces occupy those squares, which z-pattern squares are involved, and which pieces occupy the z-pattern squares.
   - First look for the strongest regularity in the activated squares themselves: repeated rank/file/edge/back-rank region, occupied piece identity, empty destination region, pawn-control diagonals, or a stable local geometry. Only after that inspect z-pattern; if z-pattern has a common attended object, that can override weaker movement/pawn-control counts.
   - Treat diagonal activation structure as positive evidence, not noise. Render the boards and test whether an Own/Opponent bishop or queen reaches the activated diagonal. A diagonal movement field, especially stronger on the Own back rank, is usually `Mov`; never use `Tgt` before layer 8, and in layer 8+ require move destinations to cover the main squares. Use `Tac` when a blocked line, exchange, or second piece is required.
   - Form a coarse board-level hypothesis before labeling: "these are Own pawn forward squares", "these are Own queen/rook/bishop reachable squares", "these are knight outpost/fork squares", "these are pieces in front of/behind a rook/bishop/queen", "these are castling corridor squares", "these are Opponent knight attacks on Own queen", etc. Then test the hypothesis against each top sample.
   - After identifying the exact squares/pieces, ask whether most top activation samples share the same chess logic: e.g. the activated piece is always attacked by a certain piece type, the z-pattern piece always protects the activated square, every activated square is a king escape square, or every pattern opens a queen/bishop/rook line. Do not label from vague impressions.
   - Policy move, target move, predicted move, and WDL/value fields when available.
   - Circuit input-edge context is mandatory when a circuit JSON is provided. For every feature being labeled, inspect upstream/contributing features in the same circuit, such as Lorsa features from the same layer and Transcoder features from the previous layer. Use the circuit `links`/input edges to identify strongest source nodes by contribution/weight, then match those source nodes back to evidence rows by `node_id`, `dictionary_name`, `feature_index`, `layer`, and `feature_type`. Read their existing interpretation, taxonomy/proposal if present, top activation samples, activated squares, z-pattern, and edge/contribution strength. Use these upstream features as evidence for what semantic signal is being passed into the current feature.
   - Before finalizing a circuit, check duplicate features across every previously completed circuit using `(dictionary_name, feature_index)` as the identity key. Reinspect the current boards and edges rather than copying blindly, but require the taxonomy to remain consistent. If current evidence genuinely contradicts the prior label, report the conflict for review instead of silently emitting two labels. Validate the completed proposal set has zero duplicate-label conflicts.
   - Do not satisfy the edge requirement with a generic sentence such as “inputs do not override the direct evidence.” State the semantic content of the strongest matched input and test it on the current boards. For a Transcoder, an upstream Lorsa `Pro` feature about Opponent-pawn control can make the Transcoder `Pro`; an upstream Lorsa relation involving a knight, bishop, rook, or queen can reveal protection, attack, exchange, or coordination hidden by sparse Transcoder activations.
   - Own/Opponent piece identities from the side to move; avoid raw white/black descriptions unless reporting a FEN-specific fact. When reading FEN directly, remember that FEN rows run from rank 8 to rank 1; evidence square index 0 is `a8`, 7 is `h8`, 56 is `a1`, and 63 is `h1`. Prefer evidence square names when available, and do not mentally flip the board.
   - In early layers, expect `Det` and `Mov` more often than high-level tactical/value labels. If activations are consistently on an Own/Opponent piece, prefer `Det`; if activations are squares that a piece can move to, pawn-forward squares, or castling squares, prefer `Mov` unless a stronger protection/capture relation is explicit.
   - For rook-related patterns, do not treat the activated square in isolation. If activations are in front of a rook, behind a piece with a rook supporting it, between king and rook, or z-pattern consistently attends to a rook, make the rook/line relation the main hypothesis: castling is usually `Mov`, rook-supported piece/capture/promotion is usually `Tac`, rook attacking an opposite-side piece can be `Cap`, and same-side rook defense can be `Pro`.
   - Apply the same "piece line / coordination" logic to knights, bishops, and queens:
     - Knight: check one-move and two-move knight geometry, forks, future outposts, and whether activated pieces are attacked by the knight. Pure reachable/outpost squares can be `Mov`; fork/capture/support context is usually `Tac`; knight attacking an opposite-side piece can be `Cap`.
     - Bishop: check diagonals, color complex, pins/skewers, bishop behind a pawn, and squares reachable by a bishop. Pure diagonal reachability is `Mov`; bishop plus queen/rook/knight coordination or pin/skewer/recapture context is `Tac`.
     - Queen: distinguish queen occupancy (`Det`) from queen movement range (`Mov`) and queen policy source/target (`Src`/`Tgt` in late layers). If activation squares are reachable by both Own queen and Own bishop/rook/knight, treat it as a multi-piece coordination hypothesis and usually `Tac`, not simple `Mov`.
   - In layer 0, avoid `Reg` unless there is a clearly board-independent routing/support pattern. Many stable layer-0 back-rank/edge patterns are `Spa`, `Mov`, `Det`, `Pro`, `Cap`, or `Tac` after checking activations and z-pattern.
   - Before choosing `Uninterpretable`, check whether a piece can repeatedly move to the activated squares, whether activated squares are pawn-control squares, castling corridor/endpoints, knight-offset patterns, one-square offsets, or stable spatial geometry.
   - For layer 3 and later, expect fewer pure `Det` features and more `Tac` or multi-relation features; inspect the common board situation, not only the activated square identity.
   - For middle and late layers, sparse activation on only a few squares is usually meaningful. Before using `Uninterpretable`, inspect whether a specific piece can reach the activated square within one or two moves, and look for capture, exchange, pin, skewer, mating threat, defense, promotion, or pawn-break motifs.
   - A feature with only a few consistently activated squares is normally not `Uninterpretable`. After checking movement, piece identity, source/target, protection, capture, spatial structure, value, and incoming edges, use a reviewable lower-confidence `Tac` for a plausible multi-piece relation rather than defaulting sparse middle/late-layer features to `Uninterpretable`. Reserve `Uninterpretable` mainly for dense or genuinely inconsistent patterns.
   - In higher layers, repeated activation on Own rook/queen, queen/bishop, or their controlled ranks/files/diagonals often represents combined major-piece control rather than two independent detectors. Prefer `Tac` when multiple piece identities or movement maps are composed; use `Det` only when occupancy alone explains the samples.
   - For the final-layer Transcoder, inspect value/WDL and top moves first, but do not force `Src` or `Tgt`. `Src`/`Tgt` can have multiple activated squares, but all main activated squares must be covered by Top Moves source/target evidence. Use `Src` only when activated squares are occupied by pieces and are Top Moves origins. Use `Tgt` only when activated squares are legal move destinations and are Top Moves destinations. If activations mix Own pieces and empty squares, or include reachable fields not covered by Top Moves, this does not satisfy `Src`/`Tgt`; at most use `Mov` unless stronger `Tac`, `Val`, `Det`, or `Uninterpretable` evidence applies.
   - For the final-layer and near-final-layer Transcoder, inspect value/WDL and top moves first, and be willing to label `Src`/`Tgt` when the move evidence is strong. `Src` can take three forms: (1) a repeated activation on one specific Own piece type, with Top Moves consistently moving that piece; (2) one activated square on each board, possibly across different Own piece types, where the matching Top Move moves the activated piece; or (3) multiple activated squares on the same board, all occupied by Own pieces, all of which appear as Top Move origins. `Tgt` has the same three structural forms, replacing origin with legal destination squares. If these structural forms are satisfied, prefer `Src`/`Tgt` over collapsing to `Mov`. Only fall back to `Mov` when the move relation is only partial or ambiguous.
   - For Transcoder features with sparse or single-square activations, do not judge only from the Transcoder activation map. If its strongest input features include a Lorsa feature with richer relational semantics, transfer that hypothesis into the Transcoder label when the top activation boards share the same relation. Example: if the Transcoder activates on/near a rook but the upstream Lorsa feature activates on the rook and attends to the piece the rook attacks, and the current samples repeatedly contain that rook-attacks-piece relation, label the Transcoder as `Cap` rather than vague `Det`/`Uninterpretable`.
   - Every fresh-label rationale must mention the connected-edge check when circuit input edges are available. Explain concretely how the strongest upstream feature supports or changes the label, or name the tested upstream relation and why the current boards contradict it. Use a form such as: "Activation alone suggests X; however input edge from L5 Lorsa #A encodes rook -> attacked piece, and the same rook-capture relation appears here, so Cap." Do not blindly copy upstream labels; require matching board positions, pieces, or top-move/value evidence in the current feature. Existing-interpretation rows are the sole exception and use only `Existing interpretation [X].`
   - Use `Uninterpretable` only when the activated region, piece relation, and move/value signal all remain genuinely unclear after the checks above. Do not use it as a default fallback for late-layer features that are simply less clean than `Src`/`Tgt` or `Tac`.
   - Render top activation boards with `scripts/render_feature_boards.py` throughout the entire circuit, including late features. Do not stop visualizing after early features. Visualization is mandatory for fresh labels that are sparse, diagonal, layer 3+, proposed `Uninterpretable`, or close `Det`/`Mov`/`Pro`/`Tac`/`Src`/`Tgt` calls. Use the visual board as inspection aid, then write the observed squares, pieces, and relations as text in the rationale.
2. Apply the taxonomy rubric conservatively.
   - `Pro` usually has two forms: a same-side piece protects/controls another square or piece, or pawn protection/control. Do not label `Pro` from pawn-control counts alone if activations are not actually all pawn-control/protection squares or if z-pattern points to a stronger tactic.
   - `Cap` is the right label when the repeated relation is opposite-side force: one side is attacking/capturing/threatening the other side's piece. If the pattern is "Opponent knight is attacking Own bishop" or similar, use `Cap`, not `Pro`.
   - In late layers, if a feature is repeatedly tied to move origins or destinations but the exact piece class varies, still consider `Src` or `Tgt` before downgrading it to `Mov`. This is especially important when the same move relation appears across several boards with one or more activated squares.
3. Emit JSONL proposals that the UI review queue can import.
4. Do not call `/circuit_taxonomy/annotate` unless the user explicitly asks to commit labels.

## Output Format

Emit one JSON object per feature:

```json
{"directory_id":"...","file_name":"...","feature_index_in_circuit":0,"dictionary_name":"...","feature_index":123,"layer":0,"feature_type":"lorsa","taxonomy":"Det","confidence":0.83,"rationale":"One-sentence reason.","evidence_summary":"Short audit trail from activations, high-z pattern, WDL/value, and move evidence."}
```

Required fields:

- `dictionary_name`
- `feature_index`
- `taxonomy`
- `confidence`
- `rationale`
- `evidence_summary`

Use taxonomy values without brackets, for example `Det`, not `[Det]`.

Rationales may be 1-3 concise sentences for easy labels, and should be longer when the label is ambiguous or when circuit input edges are available. Use the human-review style: "This looks like X, but observe/check Y, so it should be Z." It is acceptable to write rationale in Chinese when matching review notes, e.g. "这个看似是 Det，但是你看这些棋盘都有 Own doubled rooks and a protected pawn capture motif，所以是 Tac." Do not write only "evidence is insufficient" unless the mandatory geometry/move/piece checks were actually inconclusive.

Write concrete chess facts, not abstract category phrases. Avoid vague wording such as "a stronger movement or tactical relation dominates" unless it is immediately followed by the observed relation. Instead write the actual pattern: "Own knight on f6 attacks Opponent rook on h7 in most top samples", "activations are on g8/h8/f8 around the Opponent king back rank and castling corridor", "z-pattern attends from Own rook to the Opponent piece it is attacking", "activated pawn is defended by Own rook from behind and the top move is promotion", or "activated squares are all queen destinations from d1". Prefer a concrete audit trail over a short vague label: name the activated square/piece, the z-pattern square/piece, the relevant input-edge feature if any, and the repeated sample-level relation.

## Confidence

- `0.90-1.00`: direct, repeated evidence across top samples.
- `0.75-0.89`: strong pattern with minor ambiguity.
- `0.60-0.74`: plausible but needs human review.
- `<0.60`: use `Uninterpretable` unless the user asks for speculative labels.

## Scripts

Use `scripts/build_taxonomy_prompt.py` to turn raw feature-evidence JSON or JSONL into a compact prompt for another LLM/Codex pass:

```bash
python ~/.codex/skills/annotate-circuit-taxonomy/scripts/build_taxonomy_prompt.py evidence.jsonl > prompt.md
```

The script does not classify by itself; it packages evidence and rubric instructions consistently.

Use `scripts/render_feature_boards.py` when a feature needs board-level inspection, especially for layer 3+ features, `Uninterpretable` candidates, or close `Det`/`Mov`/`Tac` calls:

```bash
python ~/.codex/skills/annotate-circuit-taxonomy/scripts/render_feature_boards.py evidence.jsonl --feature 'BT4_tc_L3M_k30_e16#4814'
```

The renderer writes an HTML file with one board per top activation sample. Red squares are activations, blue squares are Lorsa source squares, yellow squares are Lorsa target squares, green arrows are top moves, and blue arrows are z-pairs. Prefer `--orientation side-to-move` so Own pieces are visually consistent. After inspection, phrase the conclusion as a textual observation, e.g. "这个看似是 Det，但是观察棋盘发现 activated pawn 总是在 Own doubled rooks 的 capture/sacrifice motif 里，所以是 Tac."
