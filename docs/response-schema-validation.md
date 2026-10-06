# AI Product Response Schema Validation

## Purpose

Validate every AI-generated credit-card concept before it is displayed or evaluated.

## Required fields

- Product name
- Target persona
- Product category
- Key features
- Value proposition
- Responsible-AI disclaimer

## Validation behaviour

- Accept responses containing all required fields.
- Reject responses with missing or invalid fields.
- Identify the affected field in the error response.
- Show the user an understandable error message.
- Log only the error code and affected field name.
- Do not log generated content, prompts, credentials or confidential information.

## Testing expectations

- Test one valid response.
- Test each required field as missing.
- Test an incorrectly formatted response.
- Confirm that invalid responses are rejected.
- Confirm that error messages do not expose sensitive information.
