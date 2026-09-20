# Measurement and filtering contract

The measurement, observation-model and tracking modules provide declared data
contracts for numerical comparisons. No physical measurements or qualified
industrial validation are supplied by this repository.

## Measurements

- the **observation mode** of every recorded comparison -- an in-surface
  distance and an ambient chord differ at second order in the perturbation,
  which is the same order as the model's own failure, so an untagged number is
  not evidence;
- the as-built coupon geometry, scanned; nominal CAD is not enough, since the
  prediction is a functional of the actual surface;
- calibrated coupon geometry and datum frame;
- commanded and measured path coordinates;
- starting-position and heading perturbations;
- tool or sensor width;
- timestamped encoder and external metrology observations;
- calibration, fixture, software, and dataset identifiers.

## Filtering

Filtering changes the observation model and is confined to the measurement and
surface-reconstruction layers. **No filter runs inside the Jacobi solver.** `engine/flows.py` and
`engine/transfer.py` integrate a declared equation at a declared step and have
no smoothing in them at all; a filter that touched them would be tuning the
model to the data through the model's own machinery.

A filter is instead part of the observation instrument, and therefore changes
the observation model. If a linear temporal filter is the matrix `F`, then

```text
y_f = F H Phi dz0 + F eta        R_f = F R F^T
```

Comparing filtered measurements against an unfiltered prediction while still
quoting the original `R` overstates the agreement by exactly the factor the
filter removed. `engine/observation_model.py` carries `TemporalFilter` and
`filtered_noise_covariance(F, R)`, `ObservationModel` holds the filter it was
built with, and `measurement.compare` raises on a trial whose
`filter_identifier` is not `"none"` unless the prediction arrives as a
`FilteredPrediction`.

**A boolean would not have been enough.** "Yes, I filtered it" is an assertion
by the caller; it cannot distinguish the right operator from a different one
wearing the same name, which is the case that silently manufactures agreement.
So the artifact carries `operator_digest` — a hash of the matrix that actually
ran — and the record carries `filter_operator_digest` for the operator that
filtered the measurement. `compare` requires identifier, version, digest *and*
causality to match, and refuses a filtered prediction against an unfiltered
trial as well, since filtering one side of a comparison biases it either way
round. Build the artifact with `apply_filter`, which applies `F`, takes the
digest from the same `F`, and returns `F R F^T` alongside the values.

### Prohibitions

- Do not tune filter parameters on validation trials.
- Do not choose smoothing strength to improve agreement with the Jacobi
  prediction. Strength is fixed from calibration data, before the trial.
- Do not use a zero-phase or future-looking smoother in an experiment claimed
  to be real time. `filter_causal` records which it was, and `offline` is an
  honest answer; a mislabelled one is not.
- Do not suppress samples near a focus because they look unstable. That is
  precisely where `rho` collapses and where the prediction is most falsifiable.
- Do not report filtered residuals with raw-sensor uncertainty.
- Do not discard outliers without recording the rule and the rejected samples.

### What each MeasurementRecord carries

`engine/measurement.py` requires, alongside the geometry and calibration
provenance: `filter_identifier`, `filter_version`, `filter_operator_digest`,
`filter_parameters`, `filter_causal`, `filter_group_delay`, `filter_tuned_on`,
`input_sampling_rate`, `output_sampling_rate`, `measurement_covariance`,
`rejected_sample_mask`, `outlier_rule` and `raw_data_digest`. A record without
them does not describe an instrument, and `MEASUREMENT_SCHEMA` is
`path-sensitivity-observation-v1`.

The record also refuses what would otherwise pass quietly: a non-finite or
negative uncertainty component, a zero combined uncertainty (which reports an
infinite signal-to-noise ratio), a non-finite or non-increasing arclength, a
non-finite separation, a non-symmetric or non-PSD measurement covariance, and
a non-positive sampling rate. `compare` reports the signal-to-noise ratio and
draws no conclusion from it; pass `resolvability_threshold` to have the
*protocol's* bar applied, since which ratio counts as resolved is a statement
about the experiment and not about the arithmetic.

### Three datasets, preserved separately

Keep these distinct records for any measurement comparison:

1. **raw observations**, as the sensor emitted them;
2. **calibrated but unfiltered** observations;
3. the **final filtered and reconstructed** trajectory, with `F`, `R_f` and the
   rejected-sample mask that produced it.

Keeping all three is what makes it possible, later, to ask whether an agreement
was in the surface or in the smoothing.

## Real time, and when the instrument knows

A sustained condition is recognised only at the end of its window. Looking back
at a finished run, acquisition began where `rho` first crossed the threshold; a
live scanner cannot say so until the window completes, because until then the
run might still be cut short. The gap is the window length, and it is real
distance travelled with the tool committed and the sensor not yet confident.

`engine/tracking.py` therefore records both times, for acquisition and for
loss:

| | retrospective | causal |
|---|---|---|
| acquisition | `acquisition_window_started_at` | `acquisition_declared_at` |
| loss | `loss_started_at` | `track_loss_declared_at` |

`AcquisitionSpec.processing` selects which drives latency, the
maximum-acquisition distance and the tracked span. This is not bookkeeping: on
the same profile, a 1.0 window and a 2.5 limit give `TRACKED` read offline and
`LATE_ACQUISITION` read causally, because the declaration lands at 3.0.

Tracked distance is always measured to `loss_started_at`, never to the
declaration — the samples between the two are degraded whether or not the
sensor had noticed. Any comparison claiming a real-time result must declare
`processing="causal"`, and an offline schedule reported as a real-time one is
the error the distinction exists to prevent.
