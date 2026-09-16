# Human Review Lessons

These notes summarize approved human review feedback from:

- `scripts/circuit_taxonomy/review-state.json`
- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits (2).jsonl`
- `scripts/circuit_taxonomy/circuit-taxonomy-evidence-exp_60ICLR_circuits_random_data_results_results_4096_k_30_e_16 (2).jsonl`

The review set contained 41 approved proposals, including 15 changed labels. The changed-label cases are especially useful for correcting systematic mistakes.

Additional approved audit set:

- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits.jsonl`
- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits (1).jsonl`
- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits (5).jsonl`
- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits (6).jsonl`
- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits (8).jsonl`
- `scripts/circuit_taxonomy/circuit-taxonomy-review-edits (10).jsonl`
- `circuit-taxonomy-review-edits (12).jsonl`
- `scripts/circuit_taxonomy/evidence/trace_0004_a1b1_64637ba35aa2.evidence.jsonl`
- `scripts/circuit_taxonomy/trace_0001_c6c5_cf0dcec5e55f.evidence.jsonl`
- `scripts/circuit_taxonomy/trace_0001_c6c5_cf0dcec5e55f.proposals.jsonl`
- `scripts/circuit_taxonomy/trace_0001_c6c5_cf0dcec5e55f.label_prompt.md`

This audit covers layer 0-2 features from `trace_0001_c6c5_cf0dcec5e55f` plus previously reviewed comparison cases. It contains 46 approved edits. Common transitions include `Uninterpretable -> Det` (8), `Reg -> Det` (5), `Reg -> Spa` (4), `Uninterpretable -> Mov` (4), `Det -> Mov` (4), `Reg -> Mov` (3), `Mov -> Det` (3), and several important `Src/Tgt -> Det/Mov/Tac` corrections.

The `(1)` audit extends this through layer 3 and contains 73 approved edits total, including 26 layer-3 edits. In layer 3, pure Det becomes less common and tactical/compositional labels increase: important layer-3 transitions include `Det -> Tac`, `Tgt -> Tac`, `Det -> Cap`, `Det -> Pro`, `Det -> Mov`, `Tgt -> Mov`, `Src -> Det`, and `Uninterpretable -> Mov/Reg/Det`.

The `(5)` audit covers all features in the same circuit: 222 approved edits over 421 evidence/proposal rows. It reinforces that later layers need different priors. Layer 13-14 Transcoder edits strongly shift toward decision features: `Src`, `Tgt`, `Val`, and `Tac`; final-layer Transcoder reviewed labels were mostly `Tgt` (5), `Src` (5), plus `Val`, `Tac`, `Det`, and only dense unclear cases as `Uninterpretable`. Across all changed labels, frequent corrections included `Uninterpretable -> Det/Reg/Mov/Tac/Src/Tgt`, `Mov -> Uninterpretable/Tac/Tgt`, `Det -> Tac/Mov/Pro`, and `Src/Tgt -> Tac`.

The `(6)` audit covers 7 layer-0 Lorsa edits in `trace_0002_c3e1_4c1245069e61`. Its main lesson is process: first find the regularity in activated squares, then inspect the z-pattern common object. Do not jump from automated `pawn_control` or movement hits directly to `Pro`, `Cap`, or `Mov`. Layer 0 should almost never be `Reg` unless there is a clear board-independent register/routing pattern.

The `(8)` audit covers 59 additional edited features from `trace_0002_c3e1_4c1245069e61`, mostly early layers. It corrects three systematic problems: early-layer labels should more often be `Det`/`Mov` than high-level categories; Own/Opponent side must be re-derived per sample; and when activations or z-pattern point to a rook, the rook/line structure should become the main hypothesis.

The `(10)` audit covers 55 edited features from `trace_0003_c3d2_62eaf3a65372`. Frequent shifts include `Det -> Pro`, `Reg -> Spa`, and several `Mov/Det -> Spa` corrections, plus two important final-layer Transcoder corrections: `Src -> Det` and `Src -> Mov`. The main new lessons are: check FEN orientation explicitly before naming pieces, and weaken the final-layer `Src`/`Tgt` prior. Final-layer features are not automatically source/target features; `Src`/`Tgt` may have multiple activated squares, but all main activated squares must be covered by Top Moves source/target evidence. Mixed activation on Own pieces and empty squares is not `Src`/`Tgt`, and is at most `Mov` unless stronger evidence supports another label.

The `(12)` audit covers 46 approved corrections from `trace_0004_a1b1_64637ba35aa2`. Its central lesson is that a Transcoder's visible activation is often only the output object, while a connected Lorsa input carries the relation that gives the object meaning. A feature that superficially looks like piece detection can therefore be `Tac` when it composes that detected piece with an upstream movement, control, protection, or relative-position signal. Conversely, an upstream label must not be copied when the same relation is absent from the current boards.

The `(17)` audit covers 81 reviewed features from `trace_0099_f8e8_7e889b8d0623`. It exposes three process failures that must not recur: visualization stopped in the later part of the circuit, edge checks became empty boilerplate, and sparse/diagonal features were over-labeled `Uninterpretable`. Apply these corrections:

- `Src` is restricted to layers 13-14. Layer-0 through layer-12 origin-looking activations must be explained as `Det`, `Mov`, `Pro`, or `Tac`; e.g. layer-0 Lorsa `#15538` marks rook-development squares and is `Mov`, not `Src`, while layer-1 Lorsa `#9759` marks the Own rook beside the Own king and is `Pro`.
- Render boards for ambiguous features across the whole circuit, not only the beginning. Diagonal activations require an explicit bishop/queen reachability check. L0 Lorsa `#8262` is an Opponent-bishop movement field, and `#8352` is an Own-bishop movement field; both are `Mov`, not `Spa`/`Uninterpretable`.
- Never write only “strongest inputs do not override.” Inspect what the input means. L1 Transcoder `#2746` inherits an Opponent-pawn-control/protection relation from an upstream Lorsa `Pro` feature and is `Pro`. L1 Transcoder `#9947` activates on an Own bishop that protects an Own knight while that knight is attacked by an Opponent bishop; the upstream Lorsa relation makes this `Pro`, not plain `Det`.
- Use upstream Lorsa semantics to interpret sparse Transcoders. If the upstream Lorsa reveals bishop/knight exchange, rook line, pawn control, or protection that is repeated on current boards, label the Transcoder with that relation (`Pro`, `Cap`, or `Tac`) rather than its superficial occupied-square pattern.
- Sparse middle/late-layer activation is normally meaningful. Before `Uninterpretable`, test piece reachability, diagonals, controlled ranks/files, exchange/protection, and incoming edges. If only a few squares activate and a plausible composed relation remains, use lower-confidence `Tac`; examples include L13 Lorsa `#5055` and L13 Transcoder `#5477`.
- High-layer activations on Own rook/queen, queen/bishop, or their controlled ranks/files/diagonals are often coordination features. Use `Tac` when multiple pieces or control maps are composed, rather than `Det` or `Uninterpretable`.
- For an existing interpretation, emit only `Existing interpretation [X].` in the rationale. Do not append newly generated board or edge boilerplate.

## Global Corrections

- Current layer-prior policy supersedes older examples below: `Src` and `Tgt` are forbidden in layers 0-7 and allowed only in layers 8-14. Historical early-layer `Src`/`Tgt` cases must now be reinterpreted as `Det`, `Mov`, `Pro`, or `Tac`. This rule also overrides an early `existing_interpretation` carrying `Src`/`Tgt`.

- First look for activation regularity, then z-pattern regularity. Human review explicitly corrected cases where the stronger pattern was "all activations are on 4th/5th rank" or "z-pattern consistently attends to Own queen/Opponent rook", while the draft over-weighted pawn-control counts.
- Spend extra tokens when needed: for each ambiguous feature, first confirm the exact activated squares and z-pattern squares, then identify what piece or empty square is there in Own/Opponent terms. Only after that compare top activation samples for a shared logic such as "all activated pieces are attacked by Own knight", "all z-pattern pieces protect the activated square", or "all positions open a queen line after a pawn push."
- Use model-relative piece names in rationales: `Own queen`, `Opponent bishop`, etc. Avoid raw color names such as white/black unless the user asks for color-specific wording. One reviewed Det case noted that "white bishop" should have been described as `Opponent bishop`.
- This Own/Opponent conversion is mandatory. Do not conclude from raw white/black. For every sample, use `side_to_move` to decide whether a piece is Own or Opponent, and write the final rationale with Own/Opponent language.
- Check FEN board orientation explicitly when deriving pieces from FEN. FEN rows are ranks 8 to 1; index 0 is `a8`, index 7 is `h8`, index 56 is `a1`, and index 63 is `h1`. Do not flip the board vertically. After locating the piece, convert it to Own/Opponent with `side_to_move`.
- Early-layer prior from `(1)`, `(5)`, and `(8)`: layers 0-2 often have simple `Det` and `Mov` features. If activation is on a repeated Own/Opponent piece, think `Det`; if activation is on pawn-forward squares, queen/rook/bishop/knight reachable squares, king escape squares, or castling geometry, think `Mov`. Do not over-promote early features to `Tac`, `Cap`, `Pro`, `Src`, or `Tgt` without a strong repeated relation.
- Rook/line patterns need special attention. If a piece is in front of a rook, a knight/pawn is supported by a rook from behind, activations lie between king and rook, or z-pattern attends to a rook, ask "what is the rook doing?" before labeling. Castling geometry is usually `Mov`; a rook-supported capture, defended promotion pawn, or piece in front of a rook is often `Tac`; a same-side rook protecting something can be `Pro`; opposite-side rook attack can be `Cap`.
- Apply the same line/coordination question to knights, bishops, and queens. For a knight, ask whether activated squares are simple knight destinations (`Mov`), fork/outpost/capture motifs (`Tac`), or enemy pieces attacked by the knight (`Cap`). For a bishop, ask whether it is pure diagonal reachability (`Mov`) or bishop-behind-pawn/pin/skewer/recapture/queen-bishop coordination (`Tac`). For a queen, separate queen occupancy (`Det`), queen movement range (`Mov`), policy source/target (`Src`/`Tgt`), and queen plus bishop/rook/knight coordination (`Tac`).
- To discover a two-piece coordination feature from top activation samples, do this explicitly: first mark the activated squares, then test which single piece can reach them; if one piece only explains part of the samples, test a second piece. If the union/intersection of two pieces' move maps repeatedly explains the high activations, and especially if z-pattern or top moves point to one of those pieces, classify as `Tac` rather than `Mov`.
- When a circuit JSON is provided with the evidence JSONL, every feature must be reviewed with its connected input edges. Look up the current node in `links`, sort incoming source nodes by contribution/weight, and inspect the corresponding upstream evidence rows. This is especially important for Transcoder features: the Transcoder activation may be a single square, while an upstream Lorsa feature may reveal the richer relation through z-pattern.
- Use upstream input features as hypotheses, not as labels to copy. For example, if a Transcoder feature activates on a rook or a square near a rook, and a strong upstream Lorsa feature activates on the rook while attending to the piece the rook attacks, then check whether the current top activation boards also share that rook-captures/rook-attacks-piece relation. If yes, `Cap` may be the right label even though the Transcoder alone looked like `Det` or `Uninterpretable`.
- For every Transcoder, treat the surface label as provisional. First identify what the Transcoder itself marks, then inspect its strongest Lorsa inputs, and finally test the combined hypothesis on every top board. Use the primitive label only when one relation explains both sources of evidence; use `Tac` when the Transcoder combines two distinct chess predicates into one feature.
- `BT4_tc_L2M_k30_e16#4532`, `Det -> Tac`, is the canonical detector-plus-control case. Surface activation is always on the Opponent pawn immediately in front of the king, but the sole strong input is Lorsa `#7624 [Mov]`, and in all six top samples the activated pawn square is reachable/controlled by an Own bishop. The semantics are therefore `Opponent pawn + Own bishop control`, not generic Opponent-pawn detection.
- `BT4_tc_L2M_k30_e16#10559`, `Det -> Tac`, is the canonical detector-plus-protection/coordination case. The Transcoder activates on an Own rook, while its Lorsa inputs carry Own-bishop movement; checking the boards shows that the activated rook is protected by the Own bishop. Treat this as rook-bishop tactical coordination rather than a plain rook detector.
- `BT4_tc_L2M_k30_e16#13896`, `Spa -> Tac`, is the canonical relative-position-plus-movement case. Its Lorsa input `#15755` encodes squares behind pawns, while the Transcoder activations are both behind those pawns and reachable by an Own rook. The conjunction `behind pawn + rook reachability` is a multi-piece tactical feature, not generic back-rank space.
- Connected inputs can also confirm a primitive label when current evidence agrees. `BT4_tc_L0M_k30_e16#11537`, `Spa -> Mov`, receives `Lorsa #6039 [Mov]`, and its activated empty back-rank squares are repeatedly reachable by an Own rook; both edge and board evidence support `Mov`. `BT4_tc_L2M_k30_e16#1636`, `Mov -> Det`, activates directly on the Opponent king and receives a detector input, so `Det` is sufficient.
- Do not inherit `Mov` from an input merely because a movement-related edge exists. Cases such as Transcoder `#488`, `#8484`, and `#1599` were corrected to `Det` because the current activation boards did not reproduce a movement field. The edge proposes the hypothesis; current sample-level geometry must verify it.
- In middle layers, an occupied activated square is not enough for `Det`. Ask what relation selects that piece: controlled by a bishop, protected from behind, in front of a rook, mutually coordinated with another rook, or occupying an attack destination. If the selecting condition is a second chess predicate supported by input edges, prefer `Tac`.
- Multi-piece co-occurrence alone can still be `Det` when no interaction is encoded. Reviewed Lorsa features `#10079` and `#11059` detect stable piece combinations such as queen+bishop, queen+rook, or bishop+knight. Promote such a feature to `Tac` only when reachability, control, protection, attack, or another relation between the pieces is repeatedly present.
- Rationale must show the connected-edge reasoning whenever circuit links are available. Good pattern: "Activation alone looks like X, but the strongest input edge from Lorsa #A encodes Y; checking the current top samples shows the same Y relation, so Z." If the input edges do not change the label, still say that the strongest connected inputs were checked and why activation/z-pattern/top-move evidence wins.
- Rationale should include more reasoning when ambiguous: name the first plausible hypothesis, what was checked, and why the final label wins. Spend more tokens when needed; the goal is a reviewable audit trail, not the shortest answer. Avoid one-line "evidence insufficient" rationales unless piece occupancy, movement reachability, pawn control/advance, castling, knight-offset, one-square-offset, and spatial checks were inconclusive.
- Avoid vague rationale like "some tactical relation", "spatial pattern", or "a stronger movement/tactical relation dominates" when the board can be inspected. A good rationale names the actual relation: which activated square/piece, which z-pattern square/piece, and what relation repeats across top samples. Write concrete observations such as "Own knight can capture/attack the Opponent rook", "activations cluster around the Opponent king back rank/castling corridor", "the activated pawn is protected by Own rook from behind", "z-pattern attends from Own rook to the Opponent piece it attacks", or "top moves repeatedly start from the activated queen".
- Do not label a stable back-rank or edge square as `Reg` until checking occupancy and chess semantics. Several reviewed Reg cases were actually Det or Mov.
- `Reg` normally means a board-edge or empty-region support pattern, usually not an early-layer feature and usually not a single occupied piece square. If the activated square is occupied by a consistent piece, prefer `Det`.
- In layer 0, `Reg` should be especially rare. Stable edge/back-rank or king-neighborhood patterns are usually `Spa`, `Mov`, `Det`, `Pro`, `Cap`, or `Tac` after activation and z-pattern inspection.
- When a feature activates on squares around the castling corridor, rook endpoint, or king-side back-rank castling structure, prefer `Mov` if castling availability or castling movement explains the pattern.
- For Lorsa, judge the relation between activated squares and high-z source squares. A same-side high-z relation is not automatically `Pro`; if the activated squares are a movement field for the high-z piece, use `Mov`.
- For queen features, distinguish "the queen itself is detected" from "squares the queen can move to". If activations are on reachable squares and high-z/source evidence points to the queen, use `Mov`; if activations are on the occupied queen square, use `Det`.
- Low confidence `Uninterpretable` should be upgraded when a simple repeated chess pattern is visible across top samples, even if source/target hit counts are weak.
- There are fewer true `Uninterpretable` cases than an automatic pass tends to produce. If a feature is hard to classify, explicitly test whether some piece always can move to the activated square, whether the activation squares are all pawn-forward or pawn-control squares, whether offsets are knight moves or one-square relations, and whether the square pattern has stable geometry.
- Source/target hits are not sufficient by themselves. If a feature activates on the Own queen/pawn/king/etc. across samples, and only some top moves originate from that square, prefer `Det` over `Src`. If activated squares are pawn-forward routes or castling geometry with incidental top-move overlap, prefer `Mov` over `Tgt`. In final-layer Transcoder, inspect top moves early, but require all main activated squares to satisfy the source/target definition.
- Starting around layer 3, do not trust a simple Det interpretation without checking the whole board context. Look for doubled rooks, a protected pawn capture/sacrifice, a passed pawn path to promotion, pieces behind or in front of pawns, Own knight attacks on the activated pawns, and Opponent pawn control over pawn-forward squares.
- In middle and late layers, sparse activations on only a few squares are usually meaningful. Before using `Uninterpretable`, check one- and two-move reachability by a specific piece, future outposts, pawn breaks, recaptures, pins, skewers, mating threats, promotion races, and defense motifs.
- In final-layer Transcoder features, inspect top moves and WDL/value first, but do not force `Src`/`Tgt`. `Src`/`Tgt` may activate multiple squares, but all main activated squares must be covered by Top Moves source/target evidence. Use `Src` only when activated squares are occupied pieces and Top Moves origins. Use `Tgt` only when activated squares are legal destinations and Top Moves destinations. If activations mix Own pieces and empty reachable squares, it is not `Src`/`Tgt`; at most use `Mov` unless stronger `Tac`, `Val`, `Det`, or `Uninterpretable` evidence applies.
- A sparse final-layer Transcoder should not be dismissed as `Uninterpretable`: first check whether all main activated squares are Top Moves sources/targets under the strict Src/Tgt definition. Conversely, dense final-layer features with no value pattern and no clear attended region can be `Uninterpretable`.
- Late-layer `Val` does not require every sample to be obviously winning or losing. A repeated endgame/material regime, such as one side up a piece or both sides reduced to rooks with a stable value pattern, can be `Val`.
- Rationale style should mirror human review notes: "这个看似是 X，但是你看/观察这些棋盘 Y，所以是 Z." Name the concrete observation, not just an abstract category boundary.

## Error Patterns From Changed Labels

Observed transitions in the reviewed set:

- `Reg -> Det`: 3 cases. Stable edge/back-rank square was actually a consistent occupied piece, often king, queen, or rook.
- `Reg -> Mov`: 2 cases. Stable square was a pawn-move destination or castling movement square.
- `Pro -> Mov`: 2 cases. Same-side relation was a queen/castling movement field, not a protection relation.
- `Cap -> Mov`: 1 case. Opposite-side attack heuristic fired, but activations lay in own queen movement squares.
- `Det -> Mov`: 1 case. King occupancy heuristic fired, but the square pattern was castling movement.
- `Mov -> Det`: 1 case. Legal-destination overlap was incidental; the activated square was consistently the own queen.
- `Uninterpretable -> Det`: 1 case. Sparse-looking squares were actually own queen detection.
- `Uninterpretable -> Mov`: 1 case. Edge pawn movement pattern was visible despite weak aggregate hits.
- `Uninterpretable -> Tgt`: 1 case. Activated squares were high-probability own queen move targets.
- `Mov -> Tac`: 1 case. Queen movement/capture evidence required an exchange or recapture motif.
- `Reg -> Spa`: 1 case. Fixed geometric relation near own back-rank bishop/queen was spatial, not a routing register.

Additional `(8)` transitions:

- `Reg -> Spa`: many early edge/corner/back-rank patterns were spatial, not register, once no piece/move relation survived.
- `Det -> Pro`: repeated same-side protection, often Opponent pawn/king/bishop/rook protecting another Opponent piece, should override simple piece occupancy.
- `Spa -> Cap` or `Uninterpretable -> Cap`: if the common logic is Opponent knight attacking Own queen, or Opponent rook attacking Own rook, this is not a vague spatial feature.
- `Src/Tgt -> Mov` in early layers: if activations are broad movement maps such as Opponent queen on back-rank movement or Own pawn forward squares, use `Mov` unless top moves strongly and specifically select the same source/target.
- `Det -> Uninterpretable`: if a proposed attack/protection relation is weak and there is no stable piece identity or geometry, do not force a label.

## Category Notes And Examples

### Det

Use `Det` when activations identify occupied piece squares, even if those squares are on the edge or back rank.

Typical reviewed cases:

- `BT4_lorsa_L0A_k30_e16#7247`, `Reg -> Det`: activated `e1=3.970058` in a sample with top activations `e1` across 6/6 samples. Human note: this is opponent king detection, not Reg.
- `BT4_tc_L0M_k30_e16#10840`, `Reg -> Det`: activated `a8=1.475541` across 6/6 samples; the square is the own queen in the reviewed positions. Stable edge square did not imply Reg.
- `BT4_tc_L0M_k30_e16#8135`, `Mov -> Det`: activated `a8=3.137818` across 6/6 samples in an endgame; reviewed as own queen on the a-file corner, despite legal-destination overlap.
- `BT4_tc_L1M_k30_e16#10758`, `Reg -> Det`: activated `h1=1.440022` across 6/6 samples; reviewed as opponent rook detection.
- `BT4_tc_L1M_k30_e16#6536`, `Uninterpretable -> Det`: activations included `e7=1.268268`, `c5=0.291705`, `f8=0.158893`; reviewed as own queen detection.
- `BT4_lorsa_L0A_k30_e16#7247`, `Reg -> Det` in `trace_0001`: top-1 activated `e1` in 6/6 samples; reviewed as `Opponent king` detection, not a back-rank register.
- `BT4_tc_L0M_k30_e16#8135`, `Mov -> Det` in `trace_0001`: top-1 activated `a8` in 6/6 samples; reviewed as an endgame Own piece detector despite edge/corner geometry and legal-move overlap.
- `BT4_tc_L1M_k30_e16#3358`, `Reg -> Det`: activated `e8` in 6/6 samples; reviewed as `Own king` detection.
- `BT4_tc_L1M_k30_e16#13537`, `Uninterpretable -> Det`: reviewed as `Own queen` detection even though automatic evidence looked sparse.
- `BT4_tc_L2M_k30_e16#1590`, `Uninterpretable -> Det`: reviewed as `Opponent pawn` detection.
- `BT4_tc_L2M_k30_e16#5632`, `Uninterpretable -> Det`: reviewed as `Opponent knight` detection in an endgame.
- `BT4_lorsa_L3A_k30_e16#2651`, `Src -> Det`: although top-source hits existed, all activations were on Own bishop squares; reviewed as Det.
- `BT4_tc_L3M_k30_e16#10623`, `Uninterpretable -> Det`: reviewed as `Opponent pawn` detection.
- `BT4_tc_L3M_k30_e16#11376`, `Uninterpretable -> Det`: reviewed as `Own knight` detection.
- `BT4_tc_L3M_k30_e16#15009`, `Uninterpretable -> Det`: reviewed as `Own queen` detection.
- `BT4_tc_L0M_k30_e16#3464`, `Uninterpretable -> Det`: reviewed as `Opponent pawn` detection; early sparse-looking activations still become Det when the same Own/Opponent piece repeats.
- `BT4_tc_L1M_k30_e16#3358`, `Reg -> Det`: reviewed as `Own king` detection; do not confuse a back-rank king square with register.
- `BT4_tc_L2M_k30_e16#3004`, `Uninterpretable -> Det`: reviewed as `Opponent bishop` detection in an endgame.

Rationale pattern: "Activated squares are consistently occupied by the Own/Opponent [piece], so this is Det despite edge/back-rank geometry."

Trace-0001 rationale pattern: "This first looks like a stable edge/back-rank square, but after converting by side-to-move the activated square is repeatedly the Own/Opponent [piece]; therefore Det is stronger than Reg/Mov/Src."

Sample-level support:

- `BT4_tc_L1M_k30_e16#15000`, `Det -> Mov` rather than Det: sample 0 `8/3K4/1k6/pr6/Q7/8/8/8 w - - 7 60` activates `e5=1.349, f5=1.341, d5=1.247, g5=1.200, h5=1.122` with top moves `a4d4`, `a4c4`; these are Own queen reachable squares from `a4`, not occupied-piece detection. Sample 1 `8/4knQ1/2P5/p2KP3/P7/7r/8/8 w - - 1 60` activates `g5=1.325, g7=1.130, g3=1.044, g8=0.821`, again along Own queen movement lines.

### Mov

Use `Mov` for movement fields, castling movement, legal destination structures, or piece-specific reachability fields. Do not collapse these into `Pro`, `Cap`, `Det`, or `Reg` just because a piece or same/opposite-side relation is visible.

Typical reviewed cases:

- `BT4_lorsa_L0A_k30_e16#7631`, `Pro -> Mov`: activated `g1=4.10915`; high-z pairs included `g1->g1=1.314622`, `g1->e1=0.720886`, `g1->f1=0.599029`. Human note: castling-related movement around king/rook, not protection.
- `BT4_tc_L1M_k30_e16#4683`, `Reg -> Mov`: activated `f1=1.915816` across 6/6 samples; top move example `e1g1`. Reviewed as castling movement.
- `BT4_tc_L0M_k30_e16#575`, `Det -> Mov`: activations included `f1=2.172146`, `e1=0.617459`, `d1=0.519802`, `g1=0.441677`; reviewed as opponent castling rook endpoint / castling movement, not king detection.
- `BT4_lorsa_L0A_k30_e16#8256`, `Cap -> Mov`: activations included `g7=17.488358`, `d4=10.154385`, `e5=6.408301`, `h2=2.778389`; high-z pairs pointed toward `h8`. Reviewed as own queen movement range attending to own queen, not capture.
- `BT4_lorsa_L1A_k30_e16#8276`, `Pro -> Mov`: activations included `e2=4.186955`, `d3=2.506153`, `b5=2.324153`, `b7=2.129236`; high-z pairs included `e2->a6=4.052129`. Reviewed as Own queen movement field, not same-side protection.
- `BT4_lorsa_L1A_k30_e16#5810`, `Uninterpretable -> Mov`: activations included `h5=2.620219`, `h6=2.517143`, `a5=1.860209`, `a6=1.755646`; high-z pairs included `h5->h7=2.568552`, `h6->h7=2.372695`, `a6->a7=1.466302`. Reviewed as own edge-pawn movement.
- `BT4_tc_L1M_k30_e16#2518`, `Reg -> Mov`: activated `a5=0.989304` across 6/6 samples; reviewed as own a-file edge pawn moving to rank 4/5.
- `BT4_lorsa_L0A_k30_e16#5813`, `Reg -> Mov` in `trace_0001`: top activations `g8/f8` and high-z relation to `h8/e8`; reviewed as short-castling movement around rook/king, not a register.
- `BT4_lorsa_L0A_k30_e16#7631`, `Pro -> Mov` in `trace_0001`: activated `g1` with high-z `g1/e1/f1`; reviewed as castling movement, not same-side protection.
- `BT4_tc_L2M_k30_e16#8062`, `Det -> Mov`: activated `c1/d1/a1`; reviewed as opponent long-castling geometry.
- `BT4_lorsa_L1A_k30_e16#394`, `Uninterpretable -> Mov`: activations such as `a5/h5/a6`; reviewed as Own pawn forward routes.
- `BT4_lorsa_L1A_k30_e16#5793`, `Det -> Mov`: activations such as `e4/f4/c4/g5` with high-z one rank behind; reviewed as Own pawn forward squares, not pawn occupancy detection.
- `BT4_tc_L1M_k30_e16#5044`, `Uninterpretable -> Mov`: activations `h5/g5/a5/e4`; reviewed as Own pawn forward positions.
- `BT4_tc_L1M_k30_e16#11862`, `Tgt -> Mov`: activations mostly on `h5/a5/a4/b5`; reviewed as Own pawn forward positions, not target-square feature.
- `BT4_tc_L1M_k30_e16#15000`, `Det -> Mov`: activations lie on Own queen reachable squares; classify as Mov rather than detecting pieces on those squares.
- `BT4_lorsa_L2A_k30_e16#6747`, `Uninterpretable -> Mov`: activations such as `g5/c5/h4`; reviewed as mostly Own pawn forward squares.
- `BT4_lorsa_L2A_k30_e16#13147`, `Src -> Mov`: high activations and high-z pairs such as `a7<-a6`, `c6<-c5`, `h6<-h5`; reviewed as pawn movement, not a source feature.
- `BT4_lorsa_L3A_k30_e16#1438`, `Tgt -> Mov`: multiple activated squares such as `b6/d6/a6/h6` are pawn destination/advance squares; Tgt usually activates on a single selected target, while this is a broader pawn movement feature.
- `BT4_lorsa_L3A_k30_e16#11060`, `Det -> Mov`: activated squares are locations an Own knight can move to while attending to the Own knight; reviewed as Mov.
- `BT4_lorsa_L3A_k30_e16#13702`, `Uninterpretable -> Mov`: same Own-knight reachable-square pattern; do not require z-pattern if board geometry is clear.
- `BT4_tc_L3M_k30_e16#1414`, `Det -> Mov`: no z-pattern, but activated squares are all reachable by an Own bishop; reviewed as Mov.
- `BT4_tc_L3M_k30_e16#1575`, `Uninterpretable -> Mov`: in endgames, activated squares were Own pawn move squares.
- `BT4_tc_L3M_k30_e16#1621`, `Tgt -> Mov`: activated squares are Own queen reachable squares; target overlap is secondary.
- `BT4_tc_L3M_k30_e16#3713`, `Uninterpretable -> Mov`: high activations mostly on Own pawn forward positions.
- `BT4_tc_L3M_k30_e16#11075`, `Det -> Mov`: activated `g1` in 6/6, but reviewed as short castling, not Own king detection.
- `BT4_tc_L3M_k30_e16#12530`, `Uninterpretable -> Mov`: activations such as `f5/b6/g5` are Own pawn forward positions.
- `BT4_tc_L3M_k30_e16#16032`, `Uninterpretable -> Mov`: activations `f8/d8` were reviewed as castling movement.
- `BT4_lorsa_L0A_k30_e16#1334`, `Det -> Mov` in `(6)`: Own king is in check or under attack, and the feature attends to squares the king may move to. This is a back-propagated king-move feature, not Own/Opponent king detection.
- `BT4_lorsa_L0A_k30_e16#6767`, `Src -> Mov` in `(8)`: z-pattern/common attention was regular while activations were less regular; reviewed as `Opponent queen` movement on the opponent back rank rather than source-square.
- `BT4_lorsa_L0A_k30_e16#7213`, `Pro -> Mov` in `(8)`: reviewed as Own rook forward-development coverage; when the z-pattern/board points to a rook, check the rook's movement lane before calling protection.
- `BT4_lorsa_L1A_k30_e16#394`, `Uninterpretable -> Mov`: Own pawn forward route.
- `BT4_tc_L1M_k30_e16#15000`, `Det -> Mov`: Own queen reachable squares, not detection of pieces on those squares.
- `BT4_tc_L2M_k30_e16#8062`, `Det -> Mov`: Opponent long-castling geometry.
- `BT4_lorsa_L3A_k30_e16#11060`, `Det -> Mov`: activated squares are Own knight destinations while z-pattern attends to Own knight; a single knight movement map explains the samples.
- `BT4_tc_L3M_k30_e16#1414`, `Det -> Mov`: even without z-pattern, activated squares are all reachable by an Own bishop; single-piece bishop reachability is Mov.
- `BT4_tc_L3M_k30_e16#1621`, `Tgt -> Mov`: activated squares are Own queen reachable squares; target overlap is secondary in early/mid layers.

Rationale pattern: "Activated squares are reachable/movement squares for Own/Opponent [piece], with high-z or move evidence pointing to that piece; classify as Mov rather than Pro/Cap/Reg."

Trace-0001 rationale pattern: "The sparse squares first look ambiguous, but they are consistently one step in front of Own pawns, in a castling corridor, or in the Own queen movement range; this supports Mov."

Sample-level support:

- `BT4_lorsa_L0A_k30_e16#5813`, `Reg -> Mov` castling/rook movement: sample 0 `3qk2r/3r2b1/bpp3pp/p1Pn4/P2Np3/NQ2B2P/1P1R1PP1/3R2K1 b k - 3 19` activates `g8=5.024, f8=4.198`; z-pattern `g8<-h8=4.262, f8<-h8=2.744, f8<-g8=1.213`. Sample 1 `Q1b1k2r/1p4bp/5pp1/2q5/1N1p4/3P2P1/4PPBP/5RK1 b k - 0 21` activates `g8=4.969, f8=3.839`; z-pattern again attends to `h8` rook and `e8/g8`. These are king-rook corridor squares, so Mov.
- `BT4_lorsa_L0A_k30_e16#1334`, `Det -> Mov` king escape/backprop: sample 0 `7k/2p2p2/2B2np1/3Pp1Qp/4P2P/1q3PP1/6K1/8 b - - 36 61` activates `h8=4.629`, z `h8<-g7=1.916, h8<-h7=0.751`; sample 1 `5r2/2r2p1k/1p2pR2/4P1Q1/2PR2Pp/p6q/5K2/8 b - - 19 135` activates `h7=4.608`, z `h7<-g7=1.088, h7<-h8=0.818, h7<-h6=0.521`. The squares are around the Own king's escape region under pressure, so Mov rather than king Det.
- `BT4_lorsa_L3A_k30_e16#11060`, `Det -> Mov` knight map: sample 0 `r3k2r/1b1pbppp/p3p3/1p2P3/3NqP2/4B1QP/PPPR2P1/5RK1 b kq - 7 17` activates `g2=3.789, f3=3.721, g5=3.020, e4=3.003, c5=2.909`, z `g2<-b7=3.716, f3<-b7=3.655, b4<-e7=2.628, g5<-e7=2.617`. Sample 1 has similar `g2/f3/e4/b4/c5` activations. These squares form knight reachable/inverse attack geometry, so a single knight movement map explains the feature.

### Src

No changed-label Src cases appeared in this review batch, but approved unchanged cases reinforce the rubric.

Typical reviewed case:

- `BT4_lorsa_L1A_k30_e16#9682`, `Src`: activated squares included `d8:2`, `h6:1`, `e6:1`, `c6:1`; source hits were 5/6 and top-source hits 4/6. Use `Src` when activations repeatedly coincide with policy move origins.

Rationale pattern: "Activated squares repeatedly match the source square of high-probability policy moves."

Important trace-0001 corrections:

- `BT4_lorsa_L1A_k30_e16#9682`, `Src -> Det`: reviewed as Own queen detection because the feature did not consistently track top moves; source overlap alone was not enough.
- `BT4_tc_L2M_k30_e16#13354`, `Src -> Det`: reviewed as Own queen detection for the same reason.
- `BT4_lorsa_L2A_k30_e16#13147`, `Src -> Mov`: reviewed as pawn movement because high-z/activation pairs tracked pawn advance geometry.
- `BT4_tc_L13M_k30_e16#7449`, `Uninterpretable -> Src`: in a high-layer Transcoder, activated major pieces were mostly the pieces appearing as top-move origins; when sparse and not dense/noisy, check top moves before using Uninterpretable.
- `BT4_tc_L13M_k30_e16#8823`, `Uninterpretable -> Src`: reviewed as a pawn-push source feature; high layers need more top-move inspection.
- `BT4_tc_L14M_k30_e16#9599`, `Det -> Src`: activated Own pawns looked like pawn detection, but many top moves were pawn pushes from those activated squares.
- `BT4_tc_L14M_k30_e16#11130`, `Uninterpretable -> Src`: many activated squares were top-move origins, so final-layer Transcoder should be Src rather than sparse Uninterpretable.
- `BT4_tc_L14M_k30_e16#12471`, `Uninterpretable -> Src`; `#13372`, `Uninterpretable -> Src`; `#14600`, `Uninterpretable -> Src`: repeated final-layer pattern where activated squares are top-move origins.
- `BT4_lorsa_L14A_k30_e16#8309`, `Mov -> Src`: activated Own queen and top moves were queen moves, so source evidence beat generic movement.
- `BT4_lorsa_L14A_k30_e16#10679`, `Uninterpretable -> Src`: activated Own pawn and top moves moved that pawn.
- Historical note: `BT4_lorsa_L0A_k30_e16#10079` was once reviewed as early `Src`, but review batch `(17)` supersedes that policy. `Src` is now restricted to layers 13-14; re-evaluate early origin-looking patterns as `Det`, `Mov`, `Pro`, or `Tac`.
- `BT4_tc_L0M_k30_e16#10880`, `Reg -> Src`; `#13486`, `Det -> Src` in `(8)`: top move origin evidence can override Reg/Det if repeated across samples.
- `BT4_tc_L14M_k30_e16#10814`, `Src -> Det` in `(10)`: final-layer Transcoder is not automatically Src. The review found no clear top move where the high-activation piece moved, so the source prior was too strong.
- `BT4_tc_L14M_k30_e16#11103`, `Src -> Mov` in `(10)`: review note says `Src` only activates on occupied squares and `Tgt` only on legal move destinations; otherwise consider `Mov`. Use this as a final-layer guardrail.

Guardrail: if activated squares are consistently an Own queen or Own pawn and only some policy moves originate there, write the hypothesis check in the rationale and prefer Det or Mov. Src may have multiple activated squares, but all main activated squares should be occupied and covered by Top Moves source evidence. If activation mixes Own pieces and empty squares, do not call it Src; at most call it Mov unless stronger non-source evidence applies.

### Tgt

Use `Tgt` when activations are move destinations, especially when the activated squares are own queen move destinations with substantial top-move probability.

Typical reviewed case:

- `BT4_tc_L1M_k30_e16#5785`, `Uninterpretable -> Tgt`: activations included `c7=1.716241`, `b6=0.218194` in one sample; aggregate activations included `d7:3`, `c7:2`, `b7:2`, `b6:1`; source hits 0/6 but target hits 3/6. Top move example `d8c7`. Reviewed as own queen target/move destination, not Uninterpretable.

Rationale pattern: "Activated squares are high-probability move destinations for Own/Opponent [piece], so target evidence dominates sparse/noisy aggregate evidence."

Important trace-0001 corrections:

- `BT4_tc_L1M_k30_e16#11862`, `Tgt -> Mov`: activations were mostly Own pawn forward squares, so pawn movement beat target overlap.
- `BT4_tc_L2M_k30_e16#14750`, `Tgt -> Tac`: reviewed as edge-pawn forward squares that are also controlled by Opponent pawns; the combined movement/control relation was tactical.
- `BT4_tc_L13M_k30_e16#3190`, `Uninterpretable -> Tgt`: high-layer pawn-push endpoint; pure Mov is less common in high Transcoder layers.
- `BT4_tc_L13M_k30_e16#5063`, `Uninterpretable -> Tgt`: another high-layer pawn-push target feature.
- `BT4_tc_L13M_k30_e16#10004`, `Uninterpretable -> Tgt`: reviewed as pawn-push target after checking top moves.
- `BT4_tc_L13M_k30_e16#15835`, `Uninterpretable -> Tgt`: activated squares were Own queen destinations appearing in top moves.
- `BT4_tc_L14M_k30_e16#2440`, `Mov -> Tgt`: activated squares were destinations of multiple top moves, not a generic movement map.
- `BT4_tc_L14M_k30_e16#2665`, `Uninterpretable -> Tgt`: pawn-push endpoints; final-layer Transcoder contains many Tgt features.
- `BT4_tc_L14M_k30_e16#7964`, `Det -> Tgt`: pawn-push endpoint beat piece detection.
- `BT4_tc_L14M_k30_e16#11714`, `Mov -> Tgt`; `#12124`, `Mov -> Tgt`: final-layer pawn-push endpoints, so Tgt beats Mov.
- `BT4_lorsa_L14A_k30_e16#14466`, `Mov -> Tgt`: for Lorsa, also inspect whether z-pattern positions are top-move starts or ends; this was target evidence from Own move threats.

Guardrail: Tgt needs activated legal destination squares plus repeated destination evidence that is not better explained by movement geometry, tactical control, or multi-piece relations. Tgt may have multiple activated squares, but all main activated squares should be covered by Top Moves target evidence. If activation mixes occupied Own pieces and empty/reachable squares, do not call it Tgt; at most call it Mov unless stronger non-target evidence applies.

### Pro

Use `Pro` only when there is a genuine same-side protection relation: one same-side piece protects or is protected by another. Same-side high-z is insufficient by itself. The two common forms are piece-to-piece/square protection and pawn protection/control.

Typical reviewed case:

- `BT4_lorsa_L1A_k30_e16#11215`, approved `Pro`: activated squares included `e3:1`, `c3:1`, `f2:1`, `e1:1`; sample relations included `e3-b1`, `f2-b1`, `d5-b1`. Human note accepted Pro because it involved an opponent queen near the opponent king attending to the opponent king.
- `BT4_lorsa_L0A_k30_e16#3608`, `Uninterpretable -> Pro` in `trace_0001`: activations such as `a4/c4/a6/c6` are spaced rather than random and correspond to Opponent pawn control/diagonal capture squares.
- `BT4_lorsa_L0A_k30_e16#14802`, `Det -> Pro`: activations such as `h5/g6/d5/f5` were reviewed as squares controlled by Own pawns, not pawn occupancy detection.
- `BT4_lorsa_L2A_k30_e16#835`, `Mov -> Pro`: activations `e6/c6/h4/f4` and pairs like `e6<-d5`, `c6<-d5` were reviewed as front pawn control squares, not simple movement.
- `BT4_lorsa_L3A_k30_e16#1858`, `Det -> Pro`: activated squares are positions Opponent pawns can protect while attending to Opponent pawns.
- `BT4_lorsa_L3A_k30_e16#14921`, `Det -> Pro`: involves two pawns, with one pawn attending to/protecting another pawn.
- `BT4_lorsa_L0A_k30_e16#1639`, `Uninterpretable -> Pro` in `(6)`: z-pattern has a common object, so it should not be Uninterpretable. The attended/activated relation is squares protected by Opponent rook while attending to Opponent rook.
- `BT4_lorsa_L0A_k30_e16#7465`, `Det -> Pro` in `(8)`: repeated same-side protection around Opponent pawns/knights, not simple detection.
- `BT4_lorsa_L0A_k30_e16#12046`, `Det -> Pro` in `(8)`: Opponent rook/king protection of Opponent pawn; same-side protection beats Det.
- `BT4_tc_L0M_k30_e16#5593`, `Det -> Pro` in `(8)`: Opponent pieces protect an Opponent knight; if the side relation is same-side, use Pro rather than Cap.

Common false positives:

- `BT4_lorsa_L0A_k30_e16#7631`, `Pro -> Mov`: castling corridor, not protection.
- `BT4_lorsa_L1A_k30_e16#8276`, `Pro -> Mov`: Own queen movement field, not protection.
- `BT4_lorsa_L0A_k30_e16#1561`, `Pro -> Tac` in `(6)`: activations were not all pawn control/protection squares. First observe the activation pattern on the 4th/5th rank, then z-pattern attention to Own queen; the stronger relation is queen-line development after a pawn push.
- `BT4_lorsa_L0A_k30_e16#1592`, `Pro -> Tac` in `(6)`: similar pattern; z-pattern points to queen after a pawn is pushed/sacrificed to open the queen route, so this is tactical rather than pure pawn protection.

Rationale pattern: "Same-side relation is protection/defense rather than a movement field; the activated and high-z squares name defender/defended pieces."

Trace-0001 rationale pattern: "The activated squares are not pieces themselves; they are controlled by Own/Opponent pawns on diagonal capture squares. That control/protection pattern supports Pro."

Sample-level support:

- `BT4_lorsa_L0A_k30_e16#3608`, `Uninterpretable -> Pro`: sample 0 `2r4r/1k6/1p1p4/3Pp1p1/PpP1PpPp/1P6/4K1P1/2R4R w - - 96 151` activates `a4=5.660, a6=4.780, c6=4.453, e6=4.437, e4=4.265`; z-pattern includes `a4<-b3=5.284, a6<-b5=3.293, e4<-d3=2.925`. Sample 1 activates `a4=5.330, c5=4.639, c4=3.886, e5=3.557`, z `a4<-b3=5.194, c5<-d4=4.249`. The activation squares are spaced diagonal pawn-control/protection squares rather than random sparse noise.

### Cap

Use `Cap` when the repeated relation is opposite-side force: one side attacks, captures, or threatens a piece of the other side. If a pattern says "this Opponent piece is attacking this Own piece" or the reverse, use `Cap`, not `Pro`.

- `BT4_lorsa_L0A_k30_e16#8256`, `Cap -> Mov`: opposite-side attack relation hits and large activations such as `g7=17.488358`, `d4=10.154385`, `e5=6.408301` were not enough for Cap; the human review identified these as squares in the own queen movement range while attending to the own queen.
- `BT4_tc_L3M_k30_e16#5363`, `Det -> Cap` in `trace_0001`: activated pawns share the property that an Own knight can capture them. This is Cap rather than pawn detection.
- `BT4_lorsa_L0A_k30_e16#1965`, `Pro -> Cap` in `(6)`: z-pattern attends to Own bishop while activations are Opponent knights; the Opponent knight is attacking the Own bishop. A clear opposite-side piece attacking another side's piece is Cap.
- `BT4_lorsa_L0A_k30_e16#2016`, `Spa -> Cap` and `#2090`, `Uninterpretable -> Cap` in `(8)`: reviewed as Opponent knight attacking Own queen. When the repeated relation is "Opponent piece is attacking Own major piece", classify Cap even if the activation pattern first looks spatial.
- `BT4_lorsa_L0A_k30_e16#10273`, `Det -> Cap` in `(8)`: repeated Opponent rook attacking Own rook; same Own/Opponent conversion matters.
- `BT4_lorsa_L0A_k30_e16#2090`, `Det -> Cap` in `(10)`: human review noted a board-orientation error. The activation was on the Own queen, not an Opponent pawn, and the Lorsa z-pattern attends to an Opponent knight that can attack/capture that queen. This is a Cap relation and a reminder to verify FEN orientation before naming pieces.

Use `Cap` only when the activated/high-z relation is actually an attack or capture relation between opposite-side pieces, not merely a queen movement field that intersects enemy-side squares.

Sample-level support:

- `BT4_lorsa_L0A_k30_e16#2016`, `Spa -> Cap`: sample 0 `r1b3k1/pnp1qppp/2p1r3/Q2pP3/3N4/NP6/P1P2PPP/3RR1K1 w - - 8 17` activates `b2=5.048`, z `b2<-a4=5.891`, with Own queen on `a5` and Opponent knight/pressure pattern around `b2/a4`. Sample 1 `1knr3r/1ppbq3/p2p4/3Pppbn/4P1p1/2P3QP/PP1N1BP1/1K1RRBN1 w - - 1 28` activates `h4=4.704`, z `h4<-g6=5.371`, again a knight attacking Own queen pattern. The decisive logic is Opponent knight pressure on Own queen, not spatial shape.
- `BT4_lorsa_L0A_k30_e16#2090`, `Uninterpretable -> Cap`: sample 0 `rnbqk2r/pp1p1ppp/4p3/8/2PNn3/6P1/PP1QPP1P/RN2KB1R w KQkq - 1 8` activates `d7=4.224`, z `d7<-e5=4.185`; sample 1 `r1b1k2r/p4ppp/1pp1p3/3p4/q1PPn3/PNQ1P3/1P3PPP/R3KB1R w KQkq - 1 13` activates `c6=4.114`, z `c6<-e5=4.026`. The common object is the Opponent knight on/around `e5` attacking Own queen positions, supporting Cap.

### Tac

Use `Tac` when a movement/capture-looking feature depends on an exchange, recapture, or other multi-piece tactical relation.

Typical reviewed case:

- `BT4_tc_L1M_k30_e16#7067`, `Mov -> Tac`: activation example `a3=1.144709`; aggregate activations included `d5:2`, `a3:1`, `d3:1`, `b2:1`. Human note: activations are on the opponent queen; own queen can capture that queen and opponent bishop can recapture, so the feature is a complex tactical exchange, not simple movement.
- `BT4_lorsa_L0A_k30_e16#5416`, `Det -> Tac` in `trace_0001`: activations on Own knight such as `d3`, with structure behind the line where an Own rook protects the knight; this is not only knight detection.
- `BT4_tc_L2M_k30_e16#14750`, `Tgt -> Tac`: edge-pawn forward squares plus Opponent pawn control over those squares; movement alone is insufficient.
- `BT4_tc_L3M_k30_e16#4814`, `Det -> Tac`: looks like Opponent pawn detection, but the shared board motif is Own doubled rooks with a rook sacrifice/capture on a protected pawn. Starting layer 3, inspect these board-level tactical commonalities.
- `BT4_tc_L3M_k30_e16#5732`, `Tgt -> Tac`: feature represents a passed pawn path toward promotion; promotion/passed-pawn race is tactical rather than a simple target feature.
- `BT4_tc_L3M_k30_e16#9594`, `Det -> Tac`: activations on Opponent bishop are tied to bishop-behind-pawn or back-rank bishop context, so pure Det is too simple.
- `BT4_tc_L3M_k30_e16#11640`, `Det -> Tac`: Opponent pawn in front of an Opponent rook; check front/back piece dependencies before labeling Det.
- `BT4_tc_L4M_k30_e16#9999`, `Uninterpretable -> Tac`: pawn push is good because the pawn can be sacrificed/recaptured to open the position.
- `BT4_tc_L4M_k30_e16#14238`, `Uninterpretable -> Tac`: Own knight, one-move knight squares, and two-move future knight positions; future outpost reasoning makes this tactical rather than Uninterpretable.
- `BT4_tc_L8M_k30_e16#6807`, `Uninterpretable -> Tac`; `#11554`, `Uninterpretable -> Tac`: Own pawn forward squares where the pawn can be exchanged; multi-piece pawn sacrifice/exchange relation.
- `BT4_lorsa_L11A_k30_e16#10574`, `Uninterpretable -> Tac`: Own pawn can advance to a square controlled by the opponent's advanced pawn.
- `BT4_tc_L11M_k30_e16#5564`, `Uninterpretable -> Tac`: Own pawn push into Opponent pawn control; movement plus control makes this Tac.
- `BT4_lorsa_L13A_k30_e16#1912`, `Uninterpretable -> Tac`: Own pawn forward squares controlled by Opponent pawns.
- `BT4_tc_L13M_k30_e16#5855`, `Uninterpretable -> Tac`: in most active boards, pushing the pawn to the activated square produces a pawn exchange where Opponent can recapture.
- `BT4_tc_L13M_k30_e16#10400`, `Det -> Tac`: pawn push relation, but activation is one square beyond the top-move pawn push; reviewed as a subtle tactic about the pawn retaining future advance.
- `BT4_tc_L13M_k30_e16#10545`, `Uninterpretable -> Tac`: activated squares look like two-step reachability by Own major/minor pieces, often color-complex bishop movement or two-step knight geometry; late-layer multi-step uncertainty should become Tac rather than Uninterpretable.
- `BT4_tc_L13M_k30_e16#11818`, `Uninterpretable -> Tac`: activations are mostly forward attacking squares or checking/mating squares; not clean enough for Mov/Tgt, but meaningful enough for Tac.
- `BT4_tc_L13M_k30_e16#14246`, `Uninterpretable -> Tac`: breakthrough squares in front of a strong Own pawn.
- `BT4_lorsa_L14A_k30_e16#2957`, `Uninterpretable -> Tac`; `#2976`, `Det -> Tac`; `#2982`, `Uninterpretable -> Tac`: pawn push followed by recapture and attention to Own/Opponent pawns.
- `BT4_lorsa_L14A_k30_e16#3166`, `Mov -> Tac`: Opponent queen threatens checks/mate; activated/attended squares require reasoning about the opponent's next mating threats.
- `BT4_lorsa_L14A_k30_e16#4073`, `Mov -> Tac`: two Own knights coordinate; not a single knight movement feature.
- `BT4_lorsa_L14A_k30_e16#13946`, `Tgt -> Tac`: activated Opponent pawn/bishop/knight in an exchange state; any clear exchange state should push toward Tac.
- `BT4_lorsa_L14A_k30_e16#14487`, `Mov -> Tac`: activated squares are reachable by Own bishop and Own queen; queen/bishop coordination is tactical.
- `BT4_tc_L14M_k30_e16#13630`, `Src -> Tac`: many top moves are promotions, and the promoting pawn is defended by a rook behind it; promotion plus rook defense is a complex tactical feature.
- `BT4_lorsa_L0A_k30_e16#1561`, `Pro -> Tac` in `(6)`: activation pattern is on the 4th/5th rank and z-pattern repeatedly attends to Own queen; not all activations are pawn-control squares. This suggests a developmental/tactical queen-line feature.
- `BT4_lorsa_L0A_k30_e16#1592`, `Pro -> Tac` in `(6)`: pawn push/sacrifice opens the route for the queen. The z-pattern commonality is decisive, so do not stop at pawn-control counts.
- `BT4_lorsa_L0A_k30_e16#1662`, `Spa -> Tac` in `(8)`: activations were not empty spatial squares but d-file Own pawn/Own knight with knight/queen support behind it; visible early development pattern should become Tac rather than Spa.
- `BT4_lorsa_L0A_k30_e16#5416`, `Det -> Tac`: the feature attends to an Own knight on a line, and the knight is protected by an Own rook behind it; rook-supported piece is a composed tactical relation.
- `BT4_tc_L3M_k30_e16#11640`, `Det -> Tac`: Opponent pawn in front of an Opponent rook; from layer 3 onward, inspect front/back piece dependencies instead of stopping at pawn detection.
- `BT4_tc_L14M_k30_e16#13630`, `Src -> Tac`: many moves are promotions, and the promoting pawn is defended by a rook behind it; promotion plus rook defense is tactical even if source evidence exists.
- `BT4_lorsa_L13A_k30_e16#14797`, `Mov -> Tac`: contribution came from a lower-level Own queen move feature, but activated squares were mostly controlled/reachable by Own bishop; the reviewed interpretation was Own queen plus bishop coordination. This is the pattern to look for when a single-piece Mov explanation feels incomplete.
- `BT4_lorsa_L13A_k30_e16#15211`, `Mov -> Tac`: top moves and z-pattern suggested queen movement, and the activated/attended squares were queen-move destinations used for defense; classify as a queen tactical coordination feature rather than plain Mov.
- `BT4_lorsa_L14A_k30_e16#14487`, `Mov -> Tac`: activated squares were reachable by both Own bishop and Own queen. The reasoning path is: first notice a movement map, then verify one piece alone does not explain the commonality; the repeated overlap of bishop and queen reachability makes it a two-piece coordination Tac feature.

Sample-level evidence for these coordination cases:

- `BT4_lorsa_L13A_k30_e16#14797`: sample 0 `3r2k1/1p2q1pp/p2b4/P2p2p1/3Nb3/2P4P/1P1Q2P1/R3R1K1 b - - 1 24` has activations `b8=5.927, c7=4.009, f4=3.566, a7=2.245, c5=1.667`, with z-pattern `b8<-c7=5.499, f4<-c7=2.843`; top moves include `d6b8`, `d6f4`. Sample 1 `4rr1k/2q3pp/1pb1p3/p3R3/3B1P2/1P4P1/2PQ3P/3R2K1 b - - 3 30` has `a8=5.466, b7=2.096` and z `a8<-b7=5.446`. Sample 2 `r2q1rk1/p1pnbpp1/1p2b2p/n1PpN3/Q2P4/2NB3P/PP3PP1/R1B1R1K1 w - - 2 15` has `b8=5.448, c7=4.078, h3=3.491`, z `b8<-c7=5.448, c7<-c7=4.104`. These samples look like a movement map, but top activations are explained by Own bishop control/reachability while the circuit contribution points to Own queen movement; use Tac for queen+bishop coordination.
- `BT4_lorsa_L13A_k30_e16#15211`: sample 0 `3r2k1/1br3pp/p2b4/2p2p2/4p2q/1PQ1P1N1/PB3PPP/2RR2K1 w - - 0 27` activates `b5=3.333, a5=0.888, c8=0.820, c4=0.758, d5=0.752`, z `b5<-c5=3.249`, with top moves `c3c4`, `d1d6`, `d1d2`. Sample 1 `r1r3k1/1p1b1p1p/1q1p1np1/p1nPp3/4Pb2/1P2BP1P/P2QB1P1/1R1N1RKN b - - 4 19` activates `a4=3.119`, z `a4<-b4=3.132`, top moves include `b6b4`. Sample 2 `1R5Q/5p1p/p3r1pk/8/q1P2P2/4P3/5PKP/8 w - - 7 42` activates `g1=3.022, c1=2.767, d1=2.212, b1=2.139`, z `g1<-f1=2.751, c1<-f1=2.532`, top moves `h8f8`, `b8g8`. The repeated clue is not just legal movement: z/top moves indicate queen movement or queen defensive relocation, so use Tac for queen tactical coordination.
- `BT4_lorsa_L14A_k30_e16#14487`: sample 0 `r2qr1k1/p2n1p1p/1pp1b1p1/3p3N/2nP1B2/P1PB1P1P/2P3P1/R1Q1R1K1 w - - 0 21` activates `f5=4.795, g4=3.015, d7=2.279, h3=1.581, e6=1.132`, z `f5<-h3=4.632, g4<-h3=2.438, d7<-h3=1.912`. Sample 1 `6k1/4b3/1r1ppn1p/q1p3p1/2P3p1/3BP2P/P1Q2P1B/3R2K1 w - - 0 32` activates `d6=4.712, f4=3.377, e5=3.267, g3=1.591`, z `d6<-g3=4.690, f4<-g3=3.206, e5<-g3=3.003`. Sample 2 `q4rk1/5ppp/b3pB2/2n1P2Q/p1Np3P/3n2P1/P4P2/R3R1K1 b - - 1 22` activates `b7=4.609, e4=3.484`, z `b7<-h1=3.975, e4<-h1=2.668`. The high squares repeatedly sit on lines/squares that Own bishop and Own queen can both influence; one-piece Mov is incomplete, so this is a Tac coordination feature.
- `BT4_lorsa_L0A_k30_e16#1662`, `Spa -> Tac`: sample 0 `r2qr1k1/1b1n1pbp/p1pp1npB/1p2p3/3PP3/P1NB1N1P/1PPQ1PP1/R3R1K1 w - - 1 13` activates `d5=1.304, e5=0.468`, z `d5<-g8=0.238, d5<-f6=0.224, d5<-c6=0.204`; sample 1 `r1b1k2r/1pqpbpp1/p1n1pn1p/6B1/3NP3/2N5/PPPQBPPP/R4RK1 w kq - 0 10` activates `d5=1.260, e5=0.340`, z `d5<-g8=0.406, d5<-c6=0.237`. These are not empty Spa squares: they are central d/e-file Own pawn/knight development squares with knight/queen support, so Tac.
- `BT4_lorsa_L0A_k30_e16#5416`, `Det -> Tac`: sample 0 `2rr2k1/p1pqbp1p/1pn3p1/8/3PNB2/2PQR3/5PPP/4R1K1 w - - 4 23` activates `e2=4.193, e1=0.809, e4=0.421`, z `e2<-e5=4.186`; sample 1 `2r2k2/2q1b1p1/r2ppn1p/pp2p3/4P3/P1PNBP2/1P1R2PP/3RQ1K1 w - - 0 27` activates `d3=4.117`, z `d3<-d6=4.117`. The activated Own knight/line is supported by Own rook behind it; this is a rook-supported tactical relation, not just knight Det.
- `BT4_tc_L3M_k30_e16#4814`, `Det -> Tac`: sample 0 `3q4/p2r1pk1/1p2p1p1/2pr3p/P3R2P/1P1P1QP1/2P1RP2/6K1 b - - 5 35` activates `d3=1.459, b3=0.150`; sample 1 `2r3k1/p1rn1ppp/1p1pp3/8/3PR3/P1PN4/1P2RPPP/6K1 b - - 4 27` activates `c3=1.427, a3=0.316`. The shared board motif is Own doubled rooks and a rook sacrifice/capture on a protected pawn, so Tac despite pawn-looking activations.
- `BT4_tc_L3M_k30_e16#11640`, `Det -> Tac`: sample 0 `6r1/2p5/p1kb3P/4p3/P3K2P/1PB4R/8/8 b - - 1 42` activates `h4=1.517`, with top move `g8g4`; sample 1 `r1b1r3/p1k5/2pqP2p/6p1/2pQ4/2p1R3/5PPP/1R4K1 w - - 2 28` activates `a2=1.478`. These are pawns/pieces in front of an Opponent rook or in rook-line dependencies; from layer 3 this front/back relation is Tac, not simple Det.

Rationale pattern: "The feature composes piece detection, capture, and recapture/exchange context; a single movement or capture primitive is insufficient."

Trace-0001 rationale pattern: "This looks like Det/Mov at first, but the activated square participates in a second relation such as protection by an Own rook or Opponent pawn control; the composed relation supports Tac."

Layer-3 rationale pattern: "这个看似是 Det，但是你看这些棋盘都有 [doubled rooks / passed pawn / piece behind pawn / protected capture] 的共性，所以是 Tac."

Middle/late-layer rationale pattern: "这个看似是 Uninterpretable/Mov，但是观察这些棋盘，激活格都是 Own pawn 推进后会被 Opponent pawn 吃回/打开局面的格子，所以这是 Tac." For uncertain two-step reachability: "看起来不是单步 Mov，但是这些格子是 Own bishop/knight 两步内能到达或形成将军/捉子的未来位置，所以后期层应标 Tac。"

### Val

Use `Val` when WDL/value/material/endgame structure explains the feature better than local square semantics. Keep the rubric conservative, but in late layers check value before dismissing dense activations.

Typical reviewed cases:

- `BT4_tc_L13M_k30_e16#7194`, `Uninterpretable -> Val`: broad activations looked ambiguous, but active boards shared a structured rook endgame/material regime where Opponent was up a piece; Value can be a repeated endgame pattern, not only a trivial win/loss.
- `BT4_lorsa_L14A_k30_e16#5891`, `Uninterpretable -> Val`: Own side was up a piece in an endgame; count material when late-layer evidence looks dense or value-like.
- `BT4_tc_L14M_k30_e16#138`, `Uninterpretable -> Val`: WDL/value was extremely high across samples; value evidence dominated sparse square interpretation.

Guardrail: do not use `Val` merely because a tactic affects value, but do inspect WDL/value and material count before labeling a dense late-layer feature `Uninterpretable`.

### Reg

No changed-label case ended as `Reg`; several false positives started as `Reg`.

Common false positives:

- `BT4_lorsa_L0A_k30_e16#7247`, `Reg -> Det`: `e1` was opponent king detection.
- `BT4_tc_L0M_k30_e16#10840`, `Reg -> Det`: `a8` was own queen detection.
- `BT4_tc_L1M_k30_e16#2518`, `Reg -> Mov`: `a5` was own a-file pawn movement.
- `BT4_tc_L1M_k30_e16#4683`, `Reg -> Mov`: `f1` was castling movement.
- `BT4_tc_L2M_k30_e16#13333`, `Uninterpretable -> Reg` in `trace_0001`: broad activations on empty board-edge/corner squares such as `h8/b1/a1/a8/b8`; reviewed as true Reg.
- `BT4_lorsa_L3A_k30_e16#1565`, `Uninterpretable -> Reg`: large activations on board edges/corners such as `a1/h1/g1/b1/a8`; reviewed as Reg.
- `BT4_tc_L3M_k30_e16#2844`, `Uninterpretable -> Reg`: repeated edge/back-rank activations including `h1/b1/a8/c1`; reviewed as Reg.
- `BT4_lorsa_L0A_k30_e16#534`, `Reg -> Spa` in `(6)`: layer 0 generally should not be Reg; this feature focused on the Own-king surrounding region, so Spa is better than a routing/register label.

Use `Reg` only after ruling out occupied-piece detection and movement semantics. A single repeated back-rank/edge square is not enough.

### Spa

Use `Spa` for stable geometric patterns that are meaningful spatially but are not piece detection, movement, source/target, protection, capture, tactic, value, or routing/register.

Typical reviewed case:

- `BT4_lorsa_L0A_k30_e16#14179`, `Reg -> Spa`: activated `b8=1.24123` across 6/6 samples; high-z examples included `b8->a2=0.216868`, `b8->d8=0.216733`, `b8->a4=0.106391`. Human note: fixed space near own back-rank bishop/queen; not tied to a specific movable piece, so Spa rather than Reg.
- `BT4_tc_L0M_k30_e16#12470`, `Reg -> Spa` in `trace_0001`: repeated `c8`/back-rank neighborhood; reviewed as a vague but stable back-rank spatial position, not piece detection.
- `BT4_tc_L0M_k30_e16#12682`, `Reg -> Spa`: repeated `f8/g8` back-rank geometry; reviewed as spatial.
- `BT4_tc_L1M_k30_e16#2958`, `Reg -> Spa`: repeated `c8/d8/b8` back-rank geometry; reviewed as spatial rather than routing register.
- `BT4_lorsa_L3A_k30_e16#4124`, `Det -> Spa`: many activated squares share Own development-position geometry; with many activations, consider Spa or Uninterpretable before Det.
- `BT4_lorsa_L0A_k30_e16#534`, `Reg -> Spa` in `(6)`: stable Own-king neighborhood/edge geometry in layer 0 is spatial, not register.
- `BT4_lorsa_L0A_k30_e16#1575`, `Cap -> Spa` in `(6)`: activations are not consistently attacked by Own pieces; they are Opponent back-rank squares, so Spa beats Cap.

Rationale pattern: "Stable geometric relation near pieces or board structure, but not an occupied-piece detector or movement field."

Trace-0001 rationale pattern: "The feature has a stable back-rank geometry, but it is not consistently an occupied piece, pawn move, castling endpoint, or policy source/target; classify as Spa rather than Reg/Uninterpretable."

### Uninterpretable

Earlier review batches had many false-negative `Uninterpretable` cases. The full-circuit `(5)` audit adds true late-layer `Uninterpretable` cases, mostly dense or unclear activations without a value pattern.

Common false negatives:

- `BT4_lorsa_L1A_k30_e16#5810`, `Uninterpretable -> Mov`: edge-pawn movement pattern visible in activations and high-z.
- `BT4_tc_L1M_k30_e16#5785`, `Uninterpretable -> Tgt`: target hits and high-probability queen target moves were enough.
- `BT4_tc_L1M_k30_e16#6536`, `Uninterpretable -> Det`: repeated own queen occupancy explained the activations.
- `BT4_lorsa_L0A_k30_e16#3608`, `Uninterpretable -> Pro`: pawn-control diagonals explained a spaced activation pattern.
- `BT4_lorsa_L1A_k30_e16#394`, `Uninterpretable -> Mov`: Own pawn forward-route squares explained sparse activations.
- `BT4_lorsa_L2A_k30_e16#6747`, `Uninterpretable -> Mov`: mostly Own pawn forward squares.
- `BT4_tc_L2M_k30_e16#1590`, `Uninterpretable -> Det`: Opponent pawn occupancy.
- `BT4_tc_L2M_k30_e16#5632`, `Uninterpretable -> Det`: Opponent knight occupancy in an endgame.
- `BT4_lorsa_L3A_k30_e16#13702`, `Uninterpretable -> Mov`: Own knight reachable squares.
- `BT4_tc_L3M_k30_e16#1575`, `Uninterpretable -> Mov`: endgame Own pawn movement squares.
- `BT4_tc_L3M_k30_e16#3713`, `Uninterpretable -> Mov`: Own pawn forward positions.
- `BT4_tc_L3M_k30_e16#12530`, `Uninterpretable -> Mov`: Own pawn forward positions.
- `BT4_tc_L3M_k30_e16#10623`, `Uninterpretable -> Det`: Opponent pawn occupancy.
- `BT4_lorsa_L3A_k30_e16#9044`, `Pro -> Uninterpretable`: activations were too dense and did not reduce to a clear control/movement relation; dense layer-3 patterns can still be Uninterpretable.
- `BT4_tc_L13M_k30_e16#12356`, `Mov -> Uninterpretable`: about half the board activated with no clear value meaning; dense high-layer features can be Uninterpretable.
- `BT4_tc_L14M_k30_e16#6503`, `Mov -> Uninterpretable`: dense activation and no value pattern; final-layer Transcoder is not automatically Mov.
- `BT4_tc_L14M_k30_e16#16317`, `Mov -> Uninterpretable`: dense activation with unclear attended/activated region and no value feature.
- `BT4_lorsa_L14A_k30_e16#2965`, `Mov -> Uninterpretable`; `#5203`, `Mov -> Uninterpretable`; `#6969`, `Mov -> Uninterpretable`; `#7628`, `Mov -> Uninterpretable`: late-layer dense/unclear Lorsa features without clear value or board relation remain Uninterpretable.
- `BT4_lorsa_L14A_k30_e16#5892`, `Mov -> Uninterpretable`: opening-like dense activation without clear semantic focus.

Use `Uninterpretable` only after checking whether noisy aggregate evidence hides a repeated piece, movement, target, or high-z relation across individual top samples.

Middle/late-layer guardrail: sparse high activations are usually not enough for `Uninterpretable`; search harder for one/two-step piece reachability, top-move source/target, pawn exchange, promotion, pin/skewer/checkmate, defense, and value/endgame patterns. Dense unclear activations with no value pattern are the main late-layer Uninterpretable cases.

Trace-0001 checklist before Uninterpretable:

- Convert every activated occupied square to Own/Opponent.
- Ask whether the activated square is one step in front of an Own pawn or a diagonal pawn-control square.
- Ask whether the activated squares are in castling corridors/endpoints.
- Ask whether a queen/rook/bishop/knight can repeatedly move to those squares.
- Ask whether the spacing is a knight offset, one-square offset, or stable back-rank/edge geometry.
- Only then use `Uninterpretable`.
