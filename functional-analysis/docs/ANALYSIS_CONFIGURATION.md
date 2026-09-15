# Analysis configuration

Scientific settings are encoded in [config/analysis.yml](../config/analysis.yml) and the scripts. The workflow uses explicit inputs and outputs rather than selective execution of console blocks or manual file combinations.

## Scientific settings

- Air temperature observations outside the environment-specific DAP windows in `config/analysis.yml` are set to missing.
- Soil temperature outliers use a Hampel filter with `k = 6` and `t0 = 1`, followed by the configured environment-specific DAP minima.
- Non-finite PTR values are missing; the upper tail PTR cutoff is three global standard deviations. PAR_TEMP uses an absolute three standard deviation cutoff.
- Drone filters in `config/analysis.yml` include the 13 sparse WIH3.2021 environment-DAP groups.
- NGRDI and PTR were selected a priori from previous analyses.
- The weather/yield correlation figure uses all-DAP FPC scores.
- The QTL genotype effect follow-up is chromosome 7 / `FPC1_NGRDI`.
- Only no-Ze prediction results are reported.
- The five DAP and seven AGDD NGRDI component counts came from exploratory prediction comparisons, not a preset 95% variance rule or nested tuning.
- The exploratory chromosome 7 annotation window excludes the broad WIH3.2020.PHP02 interval when setting its bounds. Mixed-assembly identifiers are retained and explicitly labeled; see the refinement README.
- The primary prediction compiler is the final V6 workflow, with held-out-only LOEO RMSE. Weather correlations use the matched 10,109-record cohort.

## Execution behavior

- Air and soil station FPCA use separate single-trait scripts.
- FPCA retains the number of components actually available instead of requiring manual loop restarts.
- DAP and AGDD weather workflows save every downstream CSV and model explicitly.
- The AGDD LOEO workflow writes one projected weather score file directly.
- QTL scan, downstream, and compilation branches are explicit rather than selectively sourced blocks.
- Prediction compilers select the final dated input directories from configuration.
- Diagnostic plots and object inspection statements do not control analysis state.

No console-level interactive choices are required to run the retained scripts.
