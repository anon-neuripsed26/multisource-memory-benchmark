# Output Rules — LLM Few-Shot (Single Question)

## Output Format

Each task answers ONE question for one persona. The output is a single JSON file:

```json
{
  "question": "A1",
  "answer": "10_to_19",
  "would_skip": false
}
```

## Rules

1. The `question` field must be the exact question ID assigned in the task prompt (e.g., `"A1"`, `"Ctrl2"`).
2. The `answer` field must be a valid answer label string from that question's answer space (REQUIRED, never null).
3. The `would_skip` field is a boolean indicating whether you would prefer to abstain if allowed (REQUIRED).
4. You are forced to answer even if you mark `would_skip: true` — the `answer` field must always contain a valid label.
5. Set `would_skip: true` only when evidence is genuinely too conflicting or insufficient for a confident judgment.
6. Answer labels MUST use exact strings with underscores (e.g., `"20_or_more"`, NOT `"20 or more"`).
7. The JSON must be valid and properly formatted.

## Acceptance Criteria

A task is DONE when:
- The output JSON file exists at the specified path
- It contains valid JSON matching the schema above
- The `answer` field uses a valid label from the question's answer space
- No nulls or missing questions
