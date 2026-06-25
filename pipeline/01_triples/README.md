# Stage 01: Triples

Builds Legal RAG Bench triples and retrieval-context variants.

Input: Legal RAG Bench source data from `sources.legal_rag_bench`.

Output: `pipeline/01_triples/outputs/triples.jsonl`.

Run:

```bash
uv run python pipeline/01_triples/build.py --variants
```

List variants:

```bash
uv run python pipeline/01_triples/build.py --list-variants
```

`--variants` with no value builds all paper context variants.
