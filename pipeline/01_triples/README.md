# Triples

Builds the Legal RAG Bench triples artifact used by downstream stages.

```text
outputs/triples.jsonl
```

Run the full triples stage:

```bash
uv run python pipeline/01_triples/build.py
```

By default this writes only the base Legal RAG Bench triples.

To include context variants:

```bash
uv run python pipeline/01_triples/build.py --variants
uv run python pipeline/01_triples/build.py --variants gold_surrounding,random_context
uv run python pipeline/01_triples/build.py --list-variants
```

`--variants` with no value writes base triples plus all variants. Passing comma-separated names writes base triples plus only those variants. Answerable distractor variants place the gold passage in a deterministic per-triple slot and record it in `metadata.gold_slot`. Unanswerable variants exclude the gold passage and its topic family.

Variant builders live under `variants/` and are split by method: surrounding sections, random passages, BM25 lexical retrieval, Nomic local embeddings, GloVe averaged-vector similarity, and Contriever dense retrieval.

`gold_surrounding` is the same-neighborhood baseline: it can include adjacent chunks from the same section. The `gold_plus_*_distractors` variants are the cleaner gold-plus-noise condition because they exclude the gold passage's topic family from distractors.
