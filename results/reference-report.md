# Constant-curvature reference report

Schema: `geodesic-sensitivity-report-v1`.

Claim scope: **first-order computational analysis**.

## Numerical validation

| model | K | max error in a(s) | max error in b(s) | finite/first-order gap |
| --- | ---: | ---: | ---: | ---: |
| plane | 0 | 0.000e+00 | 0.000e+00 | 5.208e-11 |
| sphere | 1 | 9.858e-11 | 4.721e-11 | 1.631e-11 |
| hyperbolic-plane | -1 | 1.652e-10 | 1.953e-10 | 2.377e-10 |

## Application examples

### positive-curvature

- Maximum possible manufacturing gap: `0.0020196`.
- Maximum possible manufacturing overlap: `0.0709964`.
- Inspection coverage reliable in the declared example: `True`.
- Maximum possible inspection coverage gap: `0`.

### negative-curvature

- Maximum possible manufacturing gap: `0.0958231`.
- Maximum possible manufacturing overlap: `0.0020204`.
- Inspection coverage reliable in the declared example: `False`.
- Maximum possible inspection coverage gap: `0.0758231`.

## Limitations

- Jacobi fields are first-order variations, not exact finite path distances.
- Gap, overlap, and coverage outputs are deterministic bounds, not probabilities.
- Machine tracking, material deformation, and sensor detection require separate models.
- A geodesic segment is not thereby established as globally shortest.
