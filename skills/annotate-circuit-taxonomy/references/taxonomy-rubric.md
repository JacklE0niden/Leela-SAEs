# Circuit Feature Taxonomy Rubric

Use this rubric to assign one label per feature. Be conservative. If the feature does not clearly satisfy a category, choose `Uninterpretable`.

## Evidence Terms

- Activated squares: squares where the feature activation is high in top activation samples.
- High-z squares: for Lorsa samples, source squares with high z-pattern contribution.
- Directional Lorsa relation: `A <- B` means activated squares satisfying predicate `A` receive strong directional contribution from high-z squares satisfying predicate `B`.
- Move evidence: policy move, predicted move, target move, or high-probability move; source square is the move's origin and target square is the destination.
- WDL/value evidence: win/draw/loss or scalar value signal associated with the position or square.
- Own/Opponent: always interpret piece side relative to `side_to_move`. A white piece can be Own in one sample and Opponent in another. Rationales should normally use `Own queen`, `Opponent pawn`, etc., not raw white/black.
- FEN orientation: FEN board rows are ranks 8 to 1. If using square indices, index 0 is `a8`, 7 is `h8`, 56 is `a1`, and 63 is `h1`. Prefer evidence square names when available. Do not flip the board vertically when converting FEN to pieces.
- Circuit input-edge evidence: upstream or contributing features in the same circuit, including their feature type, layer, existing interpretation/proposal, activation squares, Lorsa z-pattern, and edge/contribution strength. These features can explain what semantic signal is being routed into the current feature. When a circuit JSON is provided, this evidence is mandatory for every feature, not optional.
- Layer sensitivity: layer 0-2 often contain more simple detector/movement features. In early layers, `Det` and `Mov` should be high-prior labels when activations repeatedly land on a piece or on that piece's reachable/pawn-forward/castling squares. Both `Src` and `Tgt` are forbidden in layers 0-7 and allowed only in layers 8-14. By layer 3 and later, pure `Det` becomes less common and tactical/compositional features become more common. Do not stop at "activated square is a piece" in layer 3+; inspect the shared board situation and relations around that piece.
- Late-layer Transcoder sensitivity: in the final Transcoder layer, features are close to policy/value decisions, so inspect top moves and WDL/value early. Do not force `Src`/`Tgt`: they can have multiple activated squares, but all main activated squares must be covered by Top Moves source/target evidence. `Src` requires occupied activated squares that are Top Moves origins, and `Tgt` requires activated legal destinations that are Top Moves destinations. Mixed activation on Own pieces and empty reachable squares does not satisfy `Src`/`Tgt`; at most this is `Mov` unless stronger evidence supports another label.

## Decision Order

1. First determine Own/Opponent piece identities from `side_to_move`; do not reason from raw white/black alone.
2. For ambiguous features, spend extra reasoning budget on individual top activation samples. List mentally or in scratch notes: activated square -> piece/empty; z-pattern square -> piece/empty; side-to-move-relative identity; relevant top move source/target. Do not classify until the exact squares and pieces are clear.
3. First inspect activated-square regularity before using automated hit counts: repeated rank/file/edge/back-rank region, occupied piece identity, empty target region, pawn-control diagonals, pawn-forward squares, castling corridor/endpoints, knight offsets, one-square offsets, or stable spatial geometry.
4. Convert every occupied activated and z-pattern square to Own/Opponent before reasoning. If this conversion is uncertain, do not label yet. Many reviewed mistakes came from treating the raw color as fixed rather than side-to-move-relative. When using FEN, check orientation explicitly: first FEN row is rank 8, last row is rank 1.
5. For Lorsa, inspect z-pattern after the activation pattern. If z-pattern has a common attended object (for example Own queen, Opponent rook, Own bishop), that commonality can be stronger evidence than aggregate movement/pawn-control counts.
6. Inspect circuit input edges when available for every feature, especially Transcoder features. If the user provides a circuit JSON plus an evidence JSONL, read the circuit `links`, find the strongest incoming source nodes for the current node, and match those source nodes to evidence rows by `node_id`, `dictionary_name`, `feature_index`, `layer`, and `feature_type`. If a layer-5 Transcoder feature receives strong input from layer-5 Lorsa features and layer-4 Transcoder features, use those upstream features to form hypotheses about the current feature. Lorsa inputs often expose richer relations through z-patterns that the Transcoder activation alone hides.
7. For sparse Transcoder activations, test whether the strongest upstream Lorsa/Transcoder features explain the same board relation in the current top samples. For example, if the current feature only activates on a rook square, but an upstream Lorsa feature activates on the rook and attends to the piece the rook attacks, and the current boards repeatedly have that rook-attacks-piece relation, prefer `Cap` over `Det` or `Uninterpretable`.
8. Do not blindly inherit an upstream label. Require agreement between upstream semantics, current activated squares, side-to-move-relative pieces, and current top activation samples. Every rationale must mention the input-edge check when circuit links are available: either state which strongest upstream features support/change the label, or state that the connected inputs were inspected but current activation/z-pattern/top-move evidence is more decisive.
9. Check for major/minor-piece line structure early. If squares are between king and rook, near castling rook endpoints, in front of a rook/bishop/queen, behind a piece protected by a rook/bishop/queen, or z-pattern repeatedly attends to a rook/bishop/queen/knight, use that piece relation as the main hypothesis before considering generic Spa/Det.
10. Check whether top activation samples share a concrete chess logic, not just a vague category: every activated square is attacked by a certain piece type, every z-pattern piece protects/attacks the activated square, every activated square is a king escape square, every pawn move opens a queen/bishop/rook line, every square is reachable by both queen and bishop, or every pattern involves the same capture/recapture relation.
11. Write the rationale with concrete board facts. Do not say only "movement/tactical relation dominates" or "stronger chess relation" without naming what was observed. State examples such as: activated `f7/g8/h8` are around the Opponent king back rank; Own knight attacks Opponent rook; Own rook defends the promotion pawn from behind; z-pattern attends from a rook to the piece it attacks; activated squares are queen destinations from `d1`; top moves repeatedly start from the activated queen.
12. Before `Uninterpretable`, test simple hidden structure: can a consistent Own/Opponent piece move to the activated squares within one or two moves; are the squares pawn-control diagonals, pawn-forward squares, castling corridor/endpoints, knight offsets, one-square offsets, edge/corner/rank/file geometry, or repeated spatial positions near pieces?
13. In layers 0-7, start with `Det`/`Mov` priors: occupied repeated piece -> `Det`; repeated reachable squares, pawn-forward squares, queen/rook/bishop/knight movement maps, king escape squares, or castling corridors -> `Mov`. Never use `Src` or `Tgt` before layer 8; policy overlap in these layers does not override Det/Mov/Pro/Tac.
14. In layer 0, be skeptical of `Reg`: early features rarely need a routing/register explanation. Stable back-rank or edge patterns are often `Spa`, `Mov`, `Det`, `Pro`, `Cap`, or `Tac` after activation/z-pattern inspection.
15. In middle and late layers, sparse activation on a few squares is usually meaningful. Inspect the full board for capture, exchange, pin, skewer, mating threat, defense, promotion, pawn-break, rook support, and multi-piece coordination motifs before using `Uninterpretable`.
16. For final-layer Transcoder features, first check WDL/value and top moves, but keep the label evidence-based. If all main activated squares are occupied pieces and Top Moves origins, prefer `Src`; if all main activated squares are legal destinations and Top Moves destinations, prefer `Tgt`; if activation is broad but tied to coherent value/endgame/material patterns, prefer `Val`. If activations mix Own pieces and empty squares, or only partially overlap source/target evidence, do not force a decision label.
17. If evidence is still too weak, inconsistent, dense, and not value-like after these checks, label `Uninterpretable` or `Reg`.
18. If the feature directly tracks source/target squares of policy moves, prefer `Src` or `Tgt`, but only when top move evidence covers the activated squares. `Src` requires occupied origin squares; `Tgt` requires legal destination squares. If a feature activates on both occupied Own pieces and empty squares, it fails the strict Src/Tgt definition and is at most `Mov` unless stronger non-source/target evidence applies.
19. If it tracks piece identity or occupancy, prefer `Det`, but for layer 3+ first check whether the piece participates in a repeated protection, capture, promotion, doubled-rook, passed-pawn, or front/back relation; if so prefer `Tac`, `Pro`, `Cap`, or `Mov`.
20. If it tracks attack/capture/protection relations, prefer `Cap` or `Pro`, but in late layers check whether the same evidence is actually a tactical exchange/promotion/defense motif (`Tac`) or a policy source/target feature (`Src`/`Tgt`).
21. If it tracks legal or geometric movement fields, prefer `Mov`. In late Transcoder layers, promote movement evidence to `Src`/`Tgt` only when the activated occupied source or legal destination is repeatedly selected by top moves; otherwise pawn-forward paths, reachable-square fields, and broad legal movement maps remain `Mov` or become `Tac` if a tactical relation is needed.
22. If it requires a composition of multiple chess relations or a tactical motif, prefer `Tac`.
23. If it tracks value/WDL/material/advantage/king danger broadly, prefer `Val`.
24. If it is a board-position support pattern without a clear local chess concept, prefer `Reg`.
25. Use `Spa` only for spatial/geometric board patterns that are not better explained as movement, register, source, target, capture, or protection.

## Labels

### Det: Detection

Use `Det` when activated squares identify pieces by side/type or occupancy.

Typical evidence:

- Activations land on opponent knights, own rooks, kings, queens, pawns, etc.
- Interpretation says the feature detects a piece under an added condition.
- For Lorsa: activated square is one piece type and high-z square is another identity anchor, e.g. `Opp.Queen <- OwnQueen`.

Do not use `Det` for empty movement fields around a piece; those are usually `Mov`.
In layer 3+, do not use `Det` merely because activated squares are occupied. Check whether the occupied piece has a repeated tactical context such as a protected pawn, a piece behind/in front of another piece, doubled rooks, an Own knight attacking those pieces, or a passed pawn path.

### Src: Source Square

Use `Src` only in layers 8-14 when activations land on the origin square of model policy moves. Never assign `Src` in layers 0-7; reinterpret those patterns as `Det`, `Mov`, `Pro`, or `Tac` from the board relation.

Typical evidence:

- Activated square equals source of predicted/target/high-probability move across top samples.
- Interpretation names move source, from-square, piece to move, or policy origin.

Do not use `Src` for generic piece detection on a movable piece unless the move relation is explicit and repeated. `Src` can have multiple activated squares, but the activated squares should all be occupied source squares in Top Moves. If any main activated square is empty or is not covered by source evidence, it is not strict `Src`. If activations mix Own pieces and empty reachable squares, use at most `Mov` unless stronger `Tac`/`Val`/`Det` evidence applies.

For final-layer Transcoder features, weigh top moves more strongly but require the source condition: activated occupied pieces must be Top Moves origins, and multiple activated source squares are allowed only when they are all covered by Top Moves source evidence. If top moves do not clearly move the activated pieces, use `Det`, `Mov`, `Tac`, `Val`, or `Uninterpretable` as the evidence supports.

### Tgt: Target Square

Use `Tgt` only in layers 8-14 when activations land on the destination square of model policy moves. Never assign `Tgt` in layers 0-7; reinterpret early destination-looking patterns as `Mov`, or as `Pro`/`Tac` when control or tactical context is required.

Typical evidence:

- Activated square equals destination of predicted/target/high-probability move.
- Interpretation names move target, to-square, landing square, promotion square, checking square selected by policy.

Do not use `Tgt` for incidental destination overlap when activated squares are better explained as pawn-forward squares, pawn-control squares, castling movement, queen movement range, or a tactical relation. `Tgt` can have multiple activated squares, but the activated squares should all be legal Top Moves destinations. If some main activations are occupied Own pieces or empty reachable squares not covered by target evidence, it is not strict `Tgt`.

For high/final-layer Transcoder features, many pawn pushes and queen moves are decision targets. If activated legal destinations repeatedly match top-move destinations, classify as `Tgt` rather than generic `Mov`, especially for selected pawn-push endpoints and queen destination squares. If the squares are only a reachable field or a broad movement route rather than selected destinations, prefer `Mov` or `Tac`.

### Val: Value

Use `Val` when the feature encodes WDL/value/material/advantage/king-danger signals rather than a local square rule.

Typical evidence:

- Activations correlate with WDL buckets or scalar value.
- Broad board activation whose strength varies with winning/losing/drawing positions.
- Interpretation describes material balance, positional advantage, king danger, or model value.

Do not use `Val` just because a tactical feature affects value; there must be direct value/WDL evidence.

Late-layer value features can also appear as broad activation in coherent endgame/material regimes, not only as obviously winning or losing positions. Check WDL/value and material count before calling a dense late-layer feature `Uninterpretable`.

### Cap: Capture

Use `Cap` for opposite-side attack relations involving a capturable piece or capturing piece.

Typical evidence:

- Activated square is a capturable opponent piece.
- Activated square is a piece attacking an opponent piece.
- Lorsa relation links attacker and victim across opposite sides.
- One side's piece is clearly attacking/threatening a piece of the other side, even if the activated square is the attacker or the victim. For example, if activated squares are Opponent knights and z-pattern attends to Own bishops that those knights attack, use `Cap`.

If the relation is same-side defense, use `Pro`.

### Pro: Protection

Use `Pro` for same-side defense/protection relations.

Typical evidence:

- Activated square is a defended own or opponent piece.
- Activated square is the protecting piece.
- Lorsa relation links defender and defended square/piece of the same side.
- Activated squares are same-side controlled squares, including pawn-control diagonals, when the feature represents control/protection rather than piece movement.
- Two common forms: a same-side piece protects/controls another square or piece; or pawn protection/control, especially pawn diagonal control.

If the relation is an attack on an enemy piece, use `Cap`.
Do not use `Pro` merely because pawn-control counts are high. First check that activated squares are actually protection/control squares and that z-pattern is not pointing to a stronger tactic such as opening queen lines after a pawn push.

### Mov: Piece Movement

Use `Mov` for movement geometry, attack maps, reachable squares, ray alignment, knight offsets, inverse attack fields, and potential checking squares.

Typical evidence:

- Activations form legal moves/attacks/reachable squares for a piece type.
- Activations form inverse movement around an anchor, such as squares from which a knight would attack the king.
- Lorsa relation is movement field anchored by a high-z piece, e.g. `Potential OwnKnight-check squares <- Opp.King`.
- Activations mark Own pawn forward squares or edge-pawn advance routes.
- Activations mark castling corridors or rook/king endpoints when castling is available.
- Activations mark squares reachable by an Own queen/rook/bishop/knight, even if those squares are on the edge or near opposing pieces.
- Castling patterns: activations between king and rook, rook endpoints, or back-rank king/rook corridors while castling is available. If z-pattern attends to a rook/king and the squares are castling geometry, prefer `Mov`.
- Knight patterns: one-move knight destinations, inverse knight-attack squares, or simple future outpost squares can be `Mov` when only one knight relation is needed.
- Bishop/queen patterns: pure diagonal bishop reachability or queen movement range is `Mov` when a single moving piece explains the activations.

Do not use `Mov` when the feature only marks occupied pieces; that is usually `Det`.
Do not use `Mov` when the same activated squares are systematically reachable by two cooperating pieces, such as Own queen and Own bishop; that is usually `Tac` because the feature composes two movement maps.

### Tac: Complex Tactic

Use `Tac` when the feature requires multiple predicates or a named tactical motif.

Typical evidence:

- Pins, skewers, forks, discovered attacks, trapped pieces, overloaded defenders, exchange states, contested pieces.
- Composition such as movement + protection, capture + king safety, or inverse attack + defender relation.
- Lorsa relation includes both activated square and high-z source square in a multi-part chess relation.
- A piece detector plus a protection/counter-capture relation, such as Own knight behind a line and protected by an Own rook.
- Pawn-forward or edge-pawn squares plus Opponent pawn control over those squares.
- Layer 3+ shared-board motifs: Own doubled rooks plus a pawn capture/sacrifice, a pawn protected by a back-rank pawn, Opponent bishop behind a pawn or on the back rank, a passed pawn path toward promotion, or an Own knight attacking the activated pawns.
- Rook/line motifs: a piece in front of an Opponent rook, an Own knight/pawn protected by a rook behind it, a promotion pawn defended by a rook, or a rook sacrifice/capture on a protected pawn. These are usually `Tac`, not simple `Det`.
- Knight motifs: forks, future outposts that require two knight moves, a knight attacking an activated piece, or a knight plus queen/rook support pattern are usually `Tac` rather than simple `Mov`.
- Bishop/queen coordination: if activated squares are reachable by both Own bishop and Own queen, or z-pattern/top moves indicate queen movement while activations lie on bishop/queen shared lines, label `Tac`. The key discovery process is: first see a movement-looking pattern, then test whether one piece alone explains all top samples; if two pieces' reachable squares overlap systematically, it is a coordination feature.

Prefer `Tac` over `Mov`, `Cap`, or `Pro` only when a single primitive relation is insufficient.

In middle and late layers, prefer `Tac` when sparse activations require one- or two-move reasoning or a board-level tactic: pawn push that can be recaptured, pawn break/opening a position, promotion plus rook defense, queen/bishop coordination, knight future outposts, piece attacked after a forward move, check/mate threat, pin, skewer, exchange, or defense. If the exact piece is uncertain but the squares clearly form two-step bishop/knight reachability or a tactical attacking/checking pattern, use a lower-confidence `Tac` rather than `Uninterpretable`.

### Reg: Register

Use `Reg` for board-independent support patterns or likely routing/register behavior without a local chess concept.

Typical evidence:

- Consistent board-edge, corner, rank/file, empty-region, or global support pattern.
- Broad fixed activation mask not explained by source, target, piece identity, movement, capture, protection, tactic, or value.

Use `Reg` rather than `Uninterpretable` only when a stable non-chess support pattern is visible.
In layer 0, use `Reg` very rarely. If a repeated edge/back-rank pattern is relative to Own king, Opponent back rank, castling geometry, or a repeated piece/protection/capture relation, prefer the corresponding chess/spatial label.

### Spa: Spatial

Use `Spa` for spatial/geometric board patterns that are meaningful but not tied to piece identity, movement, source/target, value, register, capture, or protection.

Typical evidence:

- Center, edge, quadrant, rank/file, diagonal, color complex, or distance-to-region pattern.
- Pattern depends on square geometry, not on a specific piece relation.

If the spatial pattern is just a fixed support/routing region, use `Reg`.

### Uninterpretable

Use `Uninterpretable` when no label above is clearly supported.

Choose `Uninterpretable` when:

- Top samples disagree strongly.
- Activations are noisy or too sparse to infer a rule.
- High-z Lorsa relation is inconsistent.
- Evidence lacks move/WDL/piece/square details.
- The best label would require speculation.

Do not choose `Uninterpretable` merely because top activations are sparse. First test the candidate explanations above: occupied Own/Opponent piece detection, pawn forward/control squares, castling geometry, queen/rook/bishop movement range, knight-offset geometry, one-square offsets, and stable spatial patterns.

In middle and late layers, sparse activations are rarely truly meaningless. Use `Uninterpretable` mainly when activations are dense or half-board-like, the attended/activated region has no clear piece/move/value/tactical meaning, and WDL/value does not reveal a coherent endgame/material pattern. Dense final-layer Transcoder features without a value pattern are often `Uninterpretable`.

## Lorsa-Specific Guidance

For Lorsa, classify using both activated squares and high-z source squares.

- `Det`: activated/source squares are piece identities.
- `Mov`: activated squares are movement fields anchored by high-z pieces.
- `Cap`: activated/source relation is opposite-side attack/capture.
- `Pro`: activated/source relation is same-side defense.
- `Tac`: relation composes several primitives, such as checking-square + defender.
- `Uninterpretable`: high-z relation is not stable across samples.

## Output Discipline

Every proposal must include:

- `taxonomy`: one of `Det`, `Src`, `Tgt`, `Val`, `Cap`, `Pro`, `Mov`, `Tac`, `Reg`, `Spa`, `Uninterpretable`.
- `confidence`: numeric between 0 and 1.
- `rationale`: 1-3 concise sentences focused on the decisive category boundary, including the hypothesis checked when evidence is ambiguous.
- `evidence_summary`: compact audit trail mentioning activated squares, high-z relation when relevant, WDL/value, and move evidence when available.

Rationales should state relative sides and the test process, and should resemble the human review grammar: "This looks like X, but observe Y, so it should be Z." Chinese-style review phrasing is acceptable when useful: "这个看似是 Det，但是你看这些棋盘都有 Own doubled rooks and a protected pawn capture motif，所以是 Tac." Avoid raw "white/black" wording unless quoting a sample; convert to Own/Opponent in the conclusion.
When the label is not obvious, the rationale may be longer than usual. It should name the concrete observed squares/pieces and the repeated logic across top activation samples, for example: "激活格是 Opponent knight，z-pattern 关注 Own bishop，而且这些 Opponent knight 都在捉 Own bishop，所以是 Cap." Avoid vague statements such as "has tactical relation" without saying which piece relation was checked.
