# Stage 05: Reward Model

Trains SFT or DPO adapters on materialized preference pairs.

Input: pair JSONL with `prompt`, `chosen`, and `rejected`.

Output: LoRA adapter under `outputs/`.

Scripts:
- `prepare_training_data.py`: build the combined CJB+LRB-v2 training file
- `train_dpo.py`: DPO fine-tuning
- `train_sft.py`: SFT fine-tuning

Paper run: train Ministral-8B with DPO on
`data/training/cjb_lrb_v2_train_dpo.jsonl`.

Use the root README for runnable paper commands. Script defaults are development
defaults.
