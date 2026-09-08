# Data contracts

Initial, extensible contracts for pipeline outputs. They are not tied to a
specific CNN, XAI method, QML library, or LLM provider.

## Disease prediction

One entry per disease id in a multi-label result:

```json
{
  "disease": "pneumonia",
  "probability": 0.87,
  "positive": true
}
```

- `disease` is a registry id, not a display name.
- `probability` is a score in `[0, 1]`.
- `positive` is a pipeline flag, not a clinical diagnosis.
- The threshold that maps scores to `positive` is not defined in Phase 0.

## Analysis findings

Multiple predictions plus optional explainability metadata:

```json
{
  "predictions": [
    {
      "disease": "pneumonia",
      "probability": 0.87,
      "positive": true
    }
  ],
  "xai": {
    "method": "gradcam",
    "target": "pneumonia",
    "heatmap_available": true
  }
}
```

`gradcam` in the example is illustrative only. The XAI method has not been
selected.

## Extensibility notes

- `AnalysisFindings.xai` accepts a single object or a list so multiple
  explanations can be attached later without a new top-level schema.
- Method-specific fields go in `XAIAttachment.details`.
- `AnalysisFindings.metadata` holds optional pipeline context (for example,
  a later image id) without changing the prediction records.
- QML comparison outputs are not part of this contract yet; they can be added
  as additional structured fields when that experiment is designed.

Python types live in `src/contracts.py`.
