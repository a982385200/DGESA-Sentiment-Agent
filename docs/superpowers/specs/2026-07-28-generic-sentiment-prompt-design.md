# Generic Sentiment Prompt Design

## Goal

Clarify the shared multilingual sentiment-classification prompt so the model applies consistent positive, neutral, and negative rules and returns output that exactly matches `PredictionPayload`.

## Scope

Only the shared classification prompt in `src/sentiment_agent/prompts/prediction.py` changes. Direct, experience-assisted, translation-assisted, reflection, and zero-shot classification continue to reuse it. Translation and attribution prompts are unchanged.

## Classification rules

- `positive`: The overall message expresses praise, satisfaction, approval, gratitude, encouragement, or a favorable evaluation.
- `negative`: The overall message expresses criticism, dissatisfaction, complaint, rejection, disappointment, harm, or an unfavorable evaluation.
- `neutral`: The message is factual, descriptive, unclear, or lacks a discernible positive or negative evaluation.
- Requests and suggestions are classified from the sentiment they express about the current situation; they are not automatically neutral or negative.
- For mixed sentiment, use the dominant final evaluation. Contrast, negation, irony, and sentiment intensity must be considered.
- The model must classify the supplied text rather than the topic or isolated sentiment words.

## Output contract

The response must be exactly one JSON object:

```json
{"label":"positive","confidence":0.9,"reason":"brief core reason"}
```

It must contain exactly `label`, `confidence`, and `reason`. `label` is one of `positive`, `neutral`, or `negative`; `confidence` is between 0 and 1; and `reason` is one short sentence of no more than 20 words. No analysis, Markdown, code fences, metadata, examples, alternative answers, or text outside the JSON object is allowed.

## Tests

Unit tests will assert that the shared system prompt contains definitions for all three labels, mixed-sentiment and suggestion guidance, and the exact JSON-only output contract. Existing prompt-builder tests and the complete offline test suite must pass.
