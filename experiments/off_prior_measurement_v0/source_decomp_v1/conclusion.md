# Stage 1.4 Target-Gap Source Decomposition Conclusion

Selected conditioning:

```text
class
```

Result summary:

- Subject-specific positive classes: 1 of 4.
- Mean real-domain gap: 0.0135.
- Mean subject-specific gap: 0.0223.
- Mean natural-hard gap: nan.
- Mean DreamBooth artifact fraction: 0.8961.
- Positive subject-gap concentration: 1.0000.

Go / Pivot checks:

- Subject-specific gap is positive for at least 3 classes: False.
- Mean subject-specific gap is positive: True.
- DreamBooth artifact fraction is below 0.75: False.
- Signal is not concentrated in one class: False.

Interpretation:

- Go / pivot decision: Pivot.
- If Go: run Stage 2 forgetting correlation with source-decomposed clean gaps.
- If Pivot: revise the paper story toward real-image projection/domain alignment or redesign the measurement.
