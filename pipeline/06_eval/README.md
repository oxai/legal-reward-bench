# Stage 06: Evaluation

Scores pairwise reward accuracy.

Input: trained model or adapter, plus pair JSONL.

Output: CSV results.

Main script: `eval_dpo.py`.

Transfer helpers:
- `eval_barexam.py`
- `build_housing_pairs.py`

Scoring: CJB uses summed log probability; LRB-v2 uses length-normalised log
probability.
