# Preference

Labels candidate responses and builds chosen/rejected preference pairs.

Input:

```text
pipeline/01_dataset/outputs/candidate_responses/*.jsonl
```

Outputs:

```text
outputs/pointwise_labels__<rubric>.jsonl
outputs/preference_pairs__<rubric>.jsonl
```

Commands:

```bash
uv run python pipeline/02_preference/label_responses.py
uv run python pipeline/02_preference/build_pairs.py
```
