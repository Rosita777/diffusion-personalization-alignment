# Target-Gap Source Decomposition Conclusion

Selected conditioning:

```text
prompt_matched
```

Result summary:

- Subject-specific positive classes: 0 of 4.
- Mean real-domain gap: 0.0130.
- Mean subject-specific gap: nan.
- Mean natural-hard gap: nan.
- Mean DreamBooth artifact fraction: nan.
- DreamBooth reference rows present: False.
- Positive subject-gap concentration: 1.0000.

Go / Pivot checks:

- Subject-specific gap is positive for at least 3 classes: False.
- Mean subject-specific gap is positive: False.
- DreamBooth artifact fraction is below 0.75: False.
- Signal is not concentrated in one class: False.

Interpretation:

- Go / pivot decision: Control-only diagnosis.
- This run does not contain DreamBooth reference rows for the selected conditioning; use it only to diagnose the base-generated versus ordinary-real control gap.
