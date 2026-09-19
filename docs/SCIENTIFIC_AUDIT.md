# Scientific and reproducibility audit

This audit distinguishes the preserved published analysis from portability code. The four files in `src/original/` remain byte-for-byte unchanged and are SHA-256 checked in CI and at runtime.

## What was verified

- Every FVG is detected from completed candles A, B, and C; outcomes begin after the formation/touch bar where stated.
- Bullish and bearish geometry is directionally consistent in all four canonical scripts.
- Equality at the A/C boundary is not an FVG. The CE script additionally enforces the published one-tick minimum.
- Higher-timeframe aggregation is anchored to the 18:00 America/New_York session boundary.
- Active-contract selection is based on total trade-date volume among quarterly MNQ outrights and is not back-adjusted.
- The canonical random seeds, target/stop ordering, ambiguity convention, matching variables, and chronological split calculations were traced to their output files.
- The portability wrapper changes path constants only in a temporary AST-generated copy. Source hashes prevent an unreviewed canonical edit from running.

## Research issues retained for fidelity

These are limitations of the published scripts, not packaging defects. They were not silently changed, because doing so could change published values.

### 1. Detailed 1-minute long-horizon right-censor asymmetry

`fvg_final_fast.py` excludes control candidates near the end of the sample for its maximum horizon, while real FVG outcome arrays can still include formations with less than the full requested future window. Rolling extrema use the available suffix for those FVGs. This can make the long-horizon FVG/control populations slightly asymmetric.

Potential impact: primarily the 1,380- and 4,140-minute comparisons. Direction and magnitude require a licensed full-data corrected rerun.

### 2. “Out-of-sample” describes a chronological event split, not a sealed pipeline holdout

The scripts sort FVG parents and report the first 70% versus final 30%, but control candidates are constructed from the full time span before that split. No target outcome is used to select a control, yet this is not a strict train-on-early/deploy-on-late design.

Potential impact: the table remains a temporal robustness check, but should not be interpreted as untouched prospective validation.

### 3. Age-decay controls are survivor-zone weighted

Conditional FVG survival is evaluated by parent event. Control survival in `fvg_strength_one_tf.py` is calculated over individual control zones rather than first averaging all controls within each parent.

Potential impact: parents with more surviving controls can receive more weight. A parent-paired corrected analysis may differ.

### 4. Bootstrap clusters use ET calendar date

Several canonical bootstraps derive clusters from `timestamp.dt.date` after Eastern conversion. Bars from 18:00–23:59 ET therefore cluster with the calendar day rather than the following CME trade date used by active-contract construction.

Potential impact: point estimates are unchanged; confidence intervals and bootstrap p-values may change modestly.

### 5. Some future outcomes can span an active-contract roll

Formation events near rolls are excluded in the detailed studies, but multi-timeframe and midpoint future scans do not universally stop at the end of the formation contract block. The CE script does explicitly stop execution at its contract segment end.

Potential impact: a small number of non-CE outcomes may include a discontinuity across the concatenated, non-back-adjusted roll.

### 6. Exploratory CE inference is not independence- or multiplicity-adjusted

The CE bootstrap resamples trade rows as if independent even though zones can overlap and multiple signals can arise from the same market move. Many depth/timeframe cells were searched. The highlighted 4H 45–50% cell is therefore exploratory and not a confirmed edge.

Potential impact: reported uncertainty is likely optimistic and the strongest selected cell is subject to multiple-testing bias.

## Compatibility decision

The canonical scripts mutate arrays returned by pandas `Series.to_numpy()`. Pandas 3 can return read-only arrays for these operations, causing a runtime failure before results are produced. The reproducibility environment pins pandas 2.2.3 rather than rewriting canonical calculations. Published definitions and output values are therefore not changed by a compatibility refactor.

## What remains unverified

No licensed Databento dataset is committed. The complete 2,303,483-row published input and all nine numerical result families could not be independently regenerated during the software audit. Synthetic data verifies mechanics, paths, causal detector behavior, active-contract selection, all dashboard routes, and canonical-script startup/execution; it does not validate the frozen market statistics.
