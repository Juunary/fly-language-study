# Task definition — revision v5.1.1 (2026-09-17): spatial reference frame stipulated

This revision changes **only the wording of the task definition and review instructions**. No sentence, label, split,
template, tokenizer or model setting of `data/draft-v4.6` / protocol v5.1 changes. It resolves adjudication
`v6-B-spatial-frame` (rule amendment v5.1-ai-review-2), which found that the definition never fixed a reference frame,
so that an intrinsic reading (relative to the animals' own fronts and sides) is admissible and would change the
reference label of converse pairs in all three languages.

## Item judgment (unchanged core)

For each sentence pair, judge whether its internal propositions express the same content: 1 means mutual entailment
(A implies B AND B implies A); one-way entailment is 0. Assess roles, negation, spatial relations, and quantities
separately from linguistic naturalness.

## Spatial relations (stipulated)

- The two sentences of a pair describe **one scene seen by one fixed external viewer**. Both sentences share that viewer.
- **Horizontal terms are viewer-relative**, never relative to an animal's own front or sides:
  EN "to the left of / to the right of", "in front of / behind";
  DE "links von / rechts von", "vor / hinter";
  KO "…의 왼쪽에 / 오른쪽에", "…의 앞에 / 뒤에".
  Under the shared viewer, the converse relations hold: A is to the left of B ⇔ B is to the right of A;
  A is in front of B ⇔ B is behind A.
- **Vertical terms are gravitational**: EN "above / below", DE "über / unter", KO "…의 위에 / 아래에";
  A is above B ⇔ B is below A.
- Judgments that assume an intrinsic (animal-centred) frame are outside this definition.

## Scope

Applies identically to English, German and Korean, to the primary (declarative) rendering and to the auxiliary
outer-frame renderings, and to all splits. Review instructions (`docs/CLAUDE_REVIEW_PROMPT.md`, package
`INSTRUCTIONS.md`, issue adjudication packages) quote this section from this revision onward. Earlier review
rounds were run without the stipulation; their item judgments are reused unchanged, and the spatial audit items were
re-certified under this stipulation by a targeted per-language re-review recorded in the certification evidence.
