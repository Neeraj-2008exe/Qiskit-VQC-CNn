# PS-Q1 Feature Dictionary — QML-BIO-MITBIH

**Source:** Heartbeat feature vectors modelled on the statistics of the MIT-BIH Arrhythmia Database (PhysioNet).  
**Construction:** Each heartbeat is a 187-sample window at 360 Hz centred on the R-peak.  
The window is divided into 5 segments of 35 samples. Six statistics are extracted per segment → 30 features.

## Feature columns (30 features)

| Column | Window | Statistic |
|--------|--------|-----------|
| w1_mean | 1 (samples 0–34) | Arithmetic mean |
| w1_std  | 1 | Standard deviation |
| w1_min  | 1 | Minimum amplitude |
| w1_max  | 1 | Maximum amplitude |
| w1_skew | 1 | Fisher skewness |
| w1_kurt | 1 | Excess kurtosis |
| w2_mean–w2_kurt | 2 (samples 35–69) | Same 6 statistics |
| w3_mean–w3_kurt | 3 (samples 70–104) | Same 6 statistics |
| w4_mean–w4_kurt | 4 (samples 105–139) | Same 6 statistics |
| w5_mean–w5_kurt | 5 (samples 140–174) | Same 6 statistics |

**Units:** Amplitude values in mV. Skewness and kurtosis are dimensionless.  
**Scaling:** Features are NOT pre-scaled. Apply your own normalisation (fit on train only).

## Label column

| Value | Meaning |
|-------|---------|
| 0 | Normal heartbeat |
| 1 | Arrhythmia |

## Files

| File | Samples | Labels |
|------|---------|--------|
| ecg_train.csv | 6,000 | Yes |
| ecg_val.csv | 1,500 | Yes |
| ecg_test_features.csv | 1,500 | No — hidden. First column `sample_id` (s0 … s1499); it is an identifier, not a feature |

**Class distribution:** ~60% Normal, ~40% Arrhythmia.  
**Official metric:** Macro-averaged F1-score.
