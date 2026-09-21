# token_log.md — LLM Usage Log

## Run Configuration

| Setting | Value |
|---|---|
| LLM Provider | `none` (deterministic template) |
| Model | N/A |
| Temperature | 0 (enforced for all providers) |
| Seed | 42 (enforced where supported) |
| Cache | SHA-256 keyed disk cache at `.llm_cache/` |

## Token Usage (Full Run — LLM_PROVIDER=none)

| Metric | Value |
|---|---|
| Total candidate rows | 20 |
| Rows requiring LLM reason | 16 (No justified) |
| Rows using deterministic template | 4 (Yes / unexplained) |
| Live LLM calls | **0** (no LLM configured) |
| Cached LLM calls | **0** |
| Input tokens | **0** |
| Output tokens | **0** |
| Estimated cost | **$0.00** (local/free; stated explicitly) |

## With LLM Enabled (estimated)

If `LLM_PROVIDER=gemini` (Gemini 2.0 Flash, free tier):

| Metric | Estimate |
|---|---|
| Rows needing LLM | 16 justified rows |
| Avg prompt length | ~300 tokens |
| Avg response length | ~60 tokens |
| Total input tokens (first run) | ~4,800 |
| Total output tokens (first run) | ~960 |
| Subsequent runs (cached) | 0 live tokens |
| Cost at Gemini free tier | **$0.00** (free tier) |

If `LLM_PROVIDER=groq` (Llama3-8b, free tier):
- Same estimates, cost = **$0.00** (free tier)

## Why Temperature=0 and Seed=42

- **temperature=0** makes the LLM deterministic: same prompt always produces same output.
- **seed=42** provides an additional determinism guarantee where the API supports it (Groq).
- **Verdict decisions never depend on the LLM**: they are computed purely in `validate.py`.
  Only the plain-English *reason string* comes from the LLM. Therefore even if the LLM
  produced different wording, the `flagged` column and `matched_note_id` would be identical.

## Demonstrating Verdict Independence from LLM

```
$ python src/run.py --no-llm      # Uses template reasons
$ python src/run.py               # Uses LLM reasons (if configured)
```

The `flagged` and `matched_note_id` columns are identical in both runs.
Only the `reason` text differs in style (template vs. LLM prose).
