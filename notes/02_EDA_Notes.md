# EDA Notes (fill in as you go)

## Step 1 — Load & shape

- `df.shape` →
  (101766, 50)
  
- `df.info()` →
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 101766 entries, 0 to 101765
Data columns (total 50 columns):
 #   Column                    Non-Null Count   Dtype 
---  ------                    --------------   ----- 
 0   encounter_id              101766 non-null  int64 
 1   patient_nbr               101766 non-null  int64 
 2   race                      101766 non-null  object
 3   gender                    101766 non-null  object
 4   age                       101766 non-null  object
 5   weight                    101766 non-null  object
 6   admission_type_id         101766 non-null  int64 
 7   discharge_disposition_id  101766 non-null  int64 
 8   admission_source_id       101766 non-null  int64 
 9   time_in_hospital          101766 non-null  int64 
 10  payer_code                101766 non-null  object
 11  medical_specialty         101766 non-null  object
 12  num_lab_procedures        101766 non-null  int64 
 13  num_procedures            101766 non-null  int64 
 14  num_medications           101766 non-null  int64 
 15  number_outpatient         101766 non-null  int64 
 16  number_emergency          101766 non-null  int64 
 17  number_inpatient          101766 non-null  int64 
 18  diag_1                    101766 non-null  object
 19  diag_2                    101766 non-null  object
 20  diag_3                    101766 non-null  object
 21  number_diagnoses          101766 non-null  int64 
 22  max_glu_serum             5346 non-null    object
 23  A1Cresult                 17018 non-null   object
 24  metformin                 101766 non-null  object
 25  repaglinide               101766 non-null  object
 26  nateglinide               101766 non-null  object
 27  chlorpropamide            101766 non-null  object
 28  glimepiride               101766 non-null  object
 29  acetohexamide             101766 non-null  object
 30  glipizide                 101766 non-null  object
 31  glyburide                 101766 non-null  object
 32  tolbutamide               101766 non-null  object
 33  pioglitazone              101766 non-null  object
 34  rosiglitazone             101766 non-null  object
 35  acarbose                  101766 non-null  object
 36  miglitol                  101766 non-null  object
 37  troglitazone              101766 non-null  object
 38  tolazamide                101766 non-null  object
 39  examide                   101766 non-null  object
 40  citoglipton               101766 non-null  object
 41  insulin                   101766 non-null  object
 42  glyburide-metformin       101766 non-null  object
 43  glipizide-metformin       101766 non-null  object
 44  glimepiride-pioglitazone  101766 non-null  object
 45  metformin-rosiglitazone   101766 non-null  object
 46  metformin-pioglitazone    101766 non-null  object
 47  change                    101766 non-null  object
 48  diabetesMed               101766 non-null  object
 49  readmitted                101766 non-null  object
dtypes: int64(13), object(37)
memory usage: 38.8+ MB

- `df.isnull().sum()` (after reload with `keep_default_na=False, na_values=['?']`) →
  race 2,273 (2.2%) · weight 98,569 (96.9%) · payer_code 40,256 (39.6%) · medical_specialty 49,949 (49.1%) · diag_1 21 · diag_2 358 · diag_3 1,423 (1.4%) · max_glu_serum 0, A1Cresult 0 (both fixed — `'None'` now reads as a real category, not NaN)
  - Note: real payer_code/medical_specialty rates (39.6%/49.1%) run lower than the source paper's Table 1 figures (52%/53%) — the paper has two different missingness figures in different sections (Table 1 vs. the body text's 40%/47%, which line up closer to what we actually see here). Going with our own measured numbers as ground truth; `01_Dataset_Schema.md` corrected to match.
- weight: **DROP** — matches paper precedent (97% missing, no salvage value)
- payer_code: **DROP** — matches paper precedent ("not considered relevant to the outcome"), also keeps this project clear of payer/cost framing on purpose
- medical_specialty: **KEEP**, matching the paper's approach exactly — fill the missing 49.1% with an explicit `"Missing"` category rather than dropping the column or the rows. Rationale: missingness itself was a significant predictor in the paper's final model (coefficient +0.463, p=0.002 vs. Cardiology reference), so collapsing it away would throw out real signal, not just clean up noise.
  ```python
  df['medical_specialty'] = df['medical_specialty'].fillna('Missing')
  ```
  Note: this *is* the correct use case for `fillna()` — these are real NaNs at this point (converted from `'?'` on load), so filling them with a literal value is exactly what the method is for. Contrast with the earlier `'?'` situation, which needed `.replace()` because `'?'` wasn't NaN yet.
- expired/hospice discharge dispositions excluded from readmission denominator: **2,423 encounters excluded** (discharge_disposition_id in [11, 13, 14, 19, 20, 21]). Working set now 101,766 − 2,423 = **99,343 encounters**.

**Step 1 closed.** Weight and payer_code dropped, medical_specialty kept with a "Missing" category, `'?'` and the disguised-`'None'` issue both fixed at load, expired/hospice encounters excluded. Ready for Step 2.

## Step 2 — Target distribution

- `df['readmitted'].value_counts(normalize=True) * 100` on the real 99,343-row working set → **NO 52.9% · >30 35.7% · <30 11.4%**
- Binary collapse for the business question: `<30` = target (already what `is_readmit`/`patient_readmit` encodes), `>30` + `NO` collapse into "not readmitted within 30 days." **Baseline rate to compare every subgroup against: 11.4%.**
- Note vs. the source paper: their own subset (first-encounter-only, hospice/expired excluded, 69,984 rows) showed ~9-9.5%. Ours (99,343 rows, all encounters, same hospice/expired exclusion but *not* restricted to first encounter) runs higher at 11.4% — plausibly because repeat encounters from frequent-utilization patients pull the encounter-level rate up. Ties directly into the next check below.
- `df['patient_nbr'].nunique()` → **69,990** unique patients across 99,343 encounters → **29,353 encounters (29.6%) belong to a patient's repeat visit**, not their first. Confirms the same independence issue the source paper flagged — worth deciding by Phase 4 whether to restrict the regression to first-encounter-only (their approach) or keep all encounters and note the caveat. Not resolved yet, flagged for later.

**Step 2 closed.**

## Step 3 — Candidate columns for the business question

Shortlist, based on Step 4's reliability-checked EDA (not a guess — every one of these was checked against `n` and `margin_of_error` before being trusted):

**Confirmed strong candidates:**
1. **number_inpatient** (banded 0/1/2/3+) — the strongest single predictor found: 8.6% → 26.4%, tight margins throughout every band.
2. **number_emergency** (banded 0/1/2/3+) — second strongest: 10.7% → 25.3%. Genuinely compounds with #1 rather than duplicating it — combined top cells reach ~34%, the flagship finding of the whole analysis.
3. **time_in_hospital** (banded into `stay_band`: 1-3 / 4-6 / 7-9 / 10-14) — cleanest univariate climb (8.4%→14.8%, then plateau), and interacts with medication count specifically at the longest stays.
4. **num_medications** (banded into `med_count_band`) — clean univariate climb (~7.6%→13%), interacts with both age and length of stay.
5. **age** (pre-binned in the source data) — real signal, though messier than the others (youngest bands unreliable, non-monotonic `20-30` spike). Interacts with medication count, especially from 60+.

**Carried forward on documented precedent, not yet independently checked in Step 4:**
6. **medical_specialty** — kept, not dropped, per the source paper's finding that missingness itself was a significant predictor in their model. Our own EDA hasn't directly verified this in this dataset — worth a quick check before treating it as confirmed, or carrying it into Phase 3/4 and letting the SQL/model results confirm or deny its value.
7. **diagnosis category** (`diag_1`, grouped via the Strack et al. ICD-9 scheme) — explicitly scheduled as a Phase 3 SQL query, not evaluated in Python EDA. Included here because it's already planned, not because it's been checked yet.

**Not included / not evaluated this phase:** race, gender, admission_type_id, admission_source_id, discharge_disposition_id (beyond the hospice/expired exclusion already applied), lab/procedure counts, A1Cresult/max_glu_serum, the 24 individual medication columns, diabetesMed/change. Leaving these out of the shortlist isn't a claim that they're irrelevant — just that they weren't part of this project's EDA scope and haven't been checked against the same reliability standard as the columns above.

**Step 3 closed. Phase 2 (EDA) complete.**

---

## Polish additions (post-Phase 2, for the final report)

**Correlation heatmap** (age_numeric, time_in_hospital, num_medications, number_inpatient, number_emergency, patient_readmit):

- `number_inpatient` correlates only 0.17 with the outcome despite being the strongest predictor found via groupby — not a contradiction. Point-biserial correlation against a binary target compresses the ceiling for any predictor, strong or weak; what matters is that 0.17 is still the *highest* of the five, agreeing with the groupby ranking. Rate tables/odds ratios measure effect size; correlation measures relationships *between predictors*, which is the actually new information here.
- `time_in_hospital` × `num_medications`: **0.46** — a real, moderate relationship (longer stays → more treatment → more medications). Flag for Phase 4: these two aren't fully independent, a regression may split credit between them.
- `number_inpatient` × `number_emergency`: **0.27** — related but far from redundant, which supports (doesn't undercut) the flagship compounding finding. If these were highly correlated (0.8+), the "genuine compounding" story would be suspect.
- `age_numeric` sits near zero with everything, including a slight negative tilt with both prior-utilization measures (-0.04, -0.09) — minor curiosity, not a claim to build on.

**Effect-size summary (percentage-point swing, lowest to highest reliable band):** prior inpatient visits 17.8pp > prior emergency visits 14.6pp > length of stay 6.4pp > number of medications 5.4pp > age 2.8pp.

**Odds ratios (the report-ready framing):**

| Finding | Reference | Comparison | Odds ratio |
|---|---|---|---|
| Prior inpatient visits | 0 visits (8.6%) | 3+ visits (26.4%) | 3.82x |
| Prior emergency visits | 0 visits (10.7%) | 3+ visits (25.3%) | 2.83x |
| **Combined (flagship)** | both 0 (8.45%) | 3+ inpatient & 2+ ER (34.5%) | **5.70x** |
| Length of stay | day 1 (8.4%) | day 10 (14.8%) | 1.90x |
| Number of medications | 1-5 meds (~7.6%) | 21+ meds (~13%) | 1.82x |
| Age | 50-60 (9.8%, lowest reliable) | 80-90 (12.6%) | 1.33x |

Odds ratio formula: `odds = rate / (1 - rate)`; ratio of comparison-group odds to reference-group odds. Standard framing in the clinical readmission literature (matches how the source paper reports its own regression), and previews how Phase 4's logistic regression coefficients will read (log-odds).

**Plotted** as a horizontal bar chart (odds ratio vs. reference group, dashed line at 1 marking "no difference from baseline"), confirming visually what the table shows: the combined flagship finding (5.70x) sits well clear of every individual factor, with prior inpatient (3.82x) and prior emergency (2.83x) the next strongest, and age (1.33x) the weakest.

**LACE index note, for the write-up:** two of the three strongest predictors found independently in this analysis — length of stay and emergency visit history — correspond directly to components of the LACE index (Length of stay, Acuity, Comorbidity, Emergency visits), a validated clinical readmission risk score already used in hospital practice. Lends external credibility: the data-driven findings converge with an established real-world tool rather than an arbitrary statistical artifact.

## Step 4 — Early observations

*(free-form notes as patterns show up — this becomes raw material for the findings doc)*

**Readmission rate by age band** (`df.groupby('age')['is_readmit'].mean() * 100`, working set = 99,343):

| age | n | rate | ~95% range |
|---|---|---|---|
| 0-10 | 160 | 1.9% | 0-4% |
| 10-20 | 690 | 5.8% | 4.1-7.5% |
| 20-30 | 1,649 | 14.3% | 12.6-16.0% |
| 30-40 | 3,764 | 11.3% | 10.3-12.3% |
| 40-50 | 9,607 | 10.7% | 10.0-11.3% |
| 50-60 | 17,060 | 9.8% | 9.3-10.2% |
| 60-70 | 22,059 | 11.3% | 10.9-11.7% |
| 70-80 | 25,331 | 12.1% | 11.7-12.5% |
| 80-90 | 16,434 | 12.6% | 12.1-13.1% |
| 90-100 | 2,589 | 11.9% | 10.6-13.2% |

- `[0-10)` — **untrustworthy**, only 160 encounters / 3 readmissions. Rate can swing enormously with one more or fewer readmitted patient. Don't cite this band as a finding.
- `[10-20)` — moderately uncertain (n=690), wider range than the big bands but not absurd.
- `[20-30)` — checked and confirmed real, not a small-n artifact: n=1,649 is enough that its range (12.6-16.0%) sits clearly above the 40-70 bands even accounting for uncertainty. **Candidate finding** — worth a follow-up cut (diagnosis mix, admission type) once we're grouping by more than one variable at a time, not resolved yet.
- Overall shape (ignoring the two unreliable youngest bands): rate bottoms out around 50-60 (~9.8%), then climbs and plateaus ~12-13% from 70+, with the 20-30 spike sitting apart from that trend.
- General rule reinforced (same one from OASIS-2): always pair a rate/average with its group size before trusting it — computed via `sqrt(rate*(1-rate)/n)` for a rough standard error, ×2 for a ~95% range.
- Turned into a reusable helper, `readmit_summary(df, group_col)` — use this for every future groupby (time_in_hospital, num_medications, prior utilization, diagnosis category) instead of re-deriving the margin of error by hand each time.
- `num_medications` raw (ungrouped, 1 to 81) showed the same small-n instability as `age`'s youngest bands — several high values (72, 74, 75, 79, 81) had n of 1-3 and swung to 0% or 100%. Banded into `med_count_band` (1-5, 6-10, ... 31+, catching the sparse tail in one stable bucket) before trusting any pattern in it. Same fix as `age`, just applied to a column that wasn't pre-binned for us.
- Univariate `med_count_band` pattern: rate climbs steadily from ~7.6% (1-5 meds) to ~13% (21+ meds), then plateaus. Biggest bubble (11-15 meds, most common band) already shows the climb — not a small-n artifact.
- **Age × med_count_band interaction** (`readmit_summary(df, ['age','med_count_band'])`, filtered to n≥30): two solid candidate findings — `[40-50) × 31+ meds` at 17.1% (n=420, ±3.6pp) and `[80-90) × 26-30 meds` at 16.5% (n=838, ±2.5pp). Both notably above the general trend line and backed by enough people to trust.
- **Caveat logged, not a finding to repeat elsewhere:** `n ≥ 30` is a rough floor, not a guarantee — `[20-30) × 26-30 meds` (17.95%, n=39) looked like the top result when sorted by rate, but its margin of error is ±12pp (true range ~6-30%). Always eyeball `margin_of_error` too, not just the n filter.
- **Formula blind spot found:** `margin_of_error` reports exactly 0.00 whenever the observed rate is exactly 0% or 100% (since `rate*(1-rate)` becomes 0 in the formula), regardless of n. Saw this on `[0-10) × 6-10 meds` (0.00%, n=63) — looks maximally certain, isn't. Rough fix when this happens: treat a believable upper bound as ~3/n instead of trusting the reported 0.

**Age × med_count_band, visualized (line chart, n≥30 only, one line per med band across age):**
- `[20-30)` spike (from the univariate age check) shows up across *every* medication band at once, not just heavy-medication patients — revises the earlier framing. Whatever drives risk in that age group isn't primarily about medication count; worth a diagnosis-mix check later, not resolved yet.
- **Candidate finding for the write-up:** lines are bunched close together through the 30s-50s, then fan out from ~60 onward. By `[80-90)`, `1-5 meds` sits ~7.7% vs. `26-30 meds` at 16.5% — more than double, a gap that barely exists at younger ages. Heavy medication load matters much more for elderly patients specifically than for younger ones — a real interaction, not just two separate main effects. Directly actionable: targeted medication reconciliation / pharmacist follow-up for elderly polypharmacy patients specifically, not a blanket high-medication-count flag across all ages.
- `31+` at `[90-100)` dropping to 5.4% is the already-flagged noisy cell (n=37) — read as noise, doesn't undercut the trend above it.

---

### Methodology note — readmission rate reliability (for the end report)

**What was calculated.** For every group examined (a single age band, a single medication-count band, or an age × medication-band combination), three values were computed:
- `rate` — the percentage of encounters in that group readmitted within 30 days: (encounters with `readmitted == '<30'`) ÷ (total encounters in the group) × 100.
- `n` — the total number of encounters in the group.
- `margin_of_error` — how much that rate could plausibly vary due to sample size alone.

**The formula.**

```
margin_of_error = 1.96 × sqrt( (rate/100) × (1 − rate/100) / n ) × 100
```

This is the standard 95% confidence interval for a sample proportion (the Wald interval). `sqrt(p(1-p)/n)` is the standard error of a proportion estimated from `n` independent observations; multiplying by 1.96 converts that standard error into an approximate 95% range under a normal approximation to the binomial distribution. In plain terms: if the underlying population were sampled repeatedly, about 95% of the resulting rate estimates would fall within ± this margin of the one actually calculated.

**Why this was done.** A readmission rate computed from a handful of patients is not equally trustworthy as the same-looking rate computed from thousands. A 14% rate from 3 patients and a 14% rate from 3,000 patients represent very different amounts of evidence, but a bare percentage doesn't communicate that difference on its own. Several early groupings in this analysis — the youngest age bands, the highest raw medication counts, and sparse age × medication-band combinations — had subgroups as small as 1-3 encounters, producing rates of 0% or 100% that were not meaningful. Attaching a margin of error to every rate made this risk explicit and checkable rather than something to eyeball or, worse, miss entirely.

**A known limitation.** The Wald interval used here breaks down at the extremes: when the observed rate is exactly 0% or 100%, the formula always returns a margin of error of exactly 0, regardless of sample size, because the `rate × (1 − rate)` term becomes 0. This does not mean the estimate is certain — it's a documented weakness of this particular method at boundary values. A more robust alternative that avoids this failure mode is the Wilson score interval, worth adopting if this analysis is extended or formalized further (e.g. in the Phase 4 model).

**What the age × medication-band plot depicts.** A line chart with age band on the x-axis and readmission rate (%) on the y-axis, with one colored line per medication-count band (1-5, 6-10, ... 31+ medications). Only age × medication-band combinations with `n ≥ 30` encounters are plotted — combinations below that threshold were excluded outright rather than shown with reduced visual weight, since below that size the estimate itself isn't meaningful enough to display. Each line traces how that medication band's readmission rate moves across the age spectrum, making it possible to see whether the relationship between medication count and readmission risk holds steady across age (lines stay roughly parallel) or changes by age group (lines converge or fan apart) — i.e., whether age and medication count interact rather than acting independently.

**What it showed.** The lines sit close together through the 30s-50s age range, then fan out substantially from about age 60 onward: patients on 26-30 medications reach a readmission rate roughly double that of patients on 1-5 medications by the 80-90 age band, a gap that is much smaller or absent at younger ages. This indicates a genuine interaction between age and medication count — a high medication count is a materially stronger readmission risk factor among elderly patients specifically than it is among younger ones, rather than a flat, age-independent effect.

**Readmission rate by `time_in_hospital`** (`readmit_summary(df, 'time_in_hospital')`):

| day | n | rate | margin_of_error |
|---|---|---|---|
| 1 | 13,824 | 8.4% | ±0.46 |
| 2 | 16,891 | 10.1% | ±0.45 |
| 3 | 17,432 | 10.8% | ±0.46 |
| 4 | 13,684 | 12.0% | ±0.54 |
| 5 | 9,749 | 12.3% | ±0.65 |
| 6 | 7,355 | 12.9% | ±0.77 |
| 7 | 5,696 | 13.1% | ±0.88 |
| 8 | 4,271 | 14.6% | ±1.06 |
| 9 | 2,879 | 14.3% | ±1.28 |
| 10 | 2,262 | 14.8% | ±1.46 |
| 11 | 1,770 | 10.9% | ±1.45 |
| 12 | 1,383 | 14.0% | ±1.83 |
| 13 | 1,152 | 12.9% | ±1.94 |
| 14 | 995 | 13.5% | ±2.12 |

**Strongest, cleanest candidate finding so far:** rate climbs almost monotonically from 8.4% (day 1) to 14.8% (day 10), every point backed by thousands of encounters and a tight margin of error. Longer length of stay tracking with higher readmission risk is a well-established real-world pattern, which is a good sanity check that the data's behaving sensibly.

Days 11-14 are noisier (smaller n, and the day-11 dip to 10.9% likely reflects the multiple-comparisons effect — checking 14 sequential groups means some will wobble by chance alone even with nothing special going on). Read days 11+ as a rough plateau around 11-14%, not as a real dip-then-recover story.

**Time_in_hospital × med_count_band, first attempt (raw days, n≥30 then n≥100 filter):** still noisy at both thresholds. Worked out why algebraically: `n = 1.96² × rate × (1-rate) / margin²` — at a ~15% rate, n=100 only gets you to a ±7pp margin, and a genuinely tight ±2pp margin needs n≈1,225. Raising the floor from 30 to 100 barely helped because the real problem is 14 days × 7 medication bands = 98 possible cells, and total daily n collapses past day 8 — no filter threshold fixes data that's just too thin at that resolution.

**Fix: banded `time_in_hospital` the same way as `age`/`num_medications`** — `stay_band` = 1-3, 4-6, 7-9, 10-14. Cleaned up immediately once cells had real weight behind them.

**Candidate finding — sharper than a flat "long stay = risk" story:** from 1-3 through 7-9 days, every medication band rises together, roughly keeping relative order. But at 10-14 days the lower/mid bands (`1-5`, `6-10`, `11-15`) drop back down, while `26-30` and `31+` keep climbing — `26-30 meds` reaches 15.25%, the highest point in the whole analysis. Suggests the real risk driver isn't "long stay" or "heavy medication load" independently, but specifically their combination. Plausible (not confirmed) explanation: a long stay with few medications may reflect something like surgical recovery rather than medical complexity, while a long stay stacked with 26+ medications likely reflects genuinely sicker patients. **Recommendation for the findings doc: flag the intersection (10+ day stay AND 26+ medications) for the most intensive discharge planning, not either factor alone.**

---

### Methodology note — length-of-stay × medication interaction (for the end report)

**What was calculated.** Using the same `readmit_summary()` helper as the age analysis, readmission rate, group size (`n`), and margin of error were computed for every combination of length-of-stay and medication-count band, to test whether the two variables interact — i.e. whether the effect of medication count on readmission risk changes depending on how long a patient stayed, rather than the two acting independently.

**Why the raw daily granularity failed, and how that was diagnosed.** The first attempt grouped by the raw `time_in_hospital` value (1-14 days) crossed with medication band (7 bands), producing up to 98 possible cells. Even after filtering to `n ≥ 30`, and then to `n ≥ 100`, the resulting chart still showed implausible swings — a medication band's rate jumping by ten or more percentage points from one day to the next. Rather than treating this as unresolvable, the sample-size formula was rearranged to solve for the *n required* to reach a specific level of precision:

```
n = 1.96² × rate × (1 − rate) / margin²
```

At a readmission rate around 15%, this shows that `n = 100` only supports a margin of error of roughly ±7 percentage points — far too wide to trust — and that a genuinely tight ±2-point margin would require closer to n ≈ 1,225. Since total encounters per day fall sharply after about day 8 (from several thousand down to under 1,000 by day 14), no realistic filtering threshold could fix cells that thin once split seven ways by medication band. This confirmed the noise was a data-density problem, not a coding error or an unreasonable filter choice.

**The fix.** `time_in_hospital` was banded into four wider groups — 1-3, 4-6, 7-9, and 10-14 days — the same approach already used for `age` and `num_medications`, pooling adjacent days so each cell has enough encounters behind it to produce a stable rate. Re-running the same `n ≥ 100` filter against these wider bands produced a chart with no implausible swings.

**What the plot depicts.** A line chart with the banded length-of-stay group on the x-axis and readmission rate (%) on the y-axis, one colored line per medication-count band, restricted to combinations with at least 100 encounters. Each line traces how a given medication band's readmission rate changes as length of stay increases, making it possible to see whether the two variables move together consistently (parallel lines) or interact (lines that converge, diverge, or cross).

**What it showed.** From 1-3 days through 7-9 days, every medication band's line rises together, keeping roughly the same relative order — consistent with the two variables acting largely independently across this range. From 7-9 to 10-14 days, however, the lines diverge sharply: the lower and middle medication bands (1-5, 6-10, 11-15) turn downward, while the two heaviest bands (26-30 and 31+) continue climbing, with 26-30 medications reaching 15.25% — the highest rate observed anywhere in this analysis. This indicates that the readmission risk associated with a long hospital stay is concentrated specifically among patients who are also on a heavy medication load, rather than being a uniform effect of stay length alone.

**Prior utilization — the headline finding.** `number_inpatient` and `number_emergency` (prior visits in the year before this encounter), banded 0 / 1 / 2 / 3+ since both are heavily skewed toward zero (66.7% and 88.8% respectively at zero):

| prior_inpatient_band | n | rate | margin_of_error |
|---|---|---|---|
| 0 | 66,245 | 8.6% | ±0.21 |
| 1 | 18,984 | 13.3% | ±0.48 |
| 2 | 7,300 | 17.9% | ±0.88 |
| 3+ | 6,814 | 26.4% | ±1.05 |

| prior_emergency_band | n | rate | margin_of_error |
|---|---|---|---|
| 0 | 88,249 | 10.7% | ±0.20 |
| 1 | 7,474 | 14.7% | ±0.80 |
| 2 | 1,984 | 18.8% | ±1.72 |
| 3+ | 1,636 | 25.3% | ±2.11 |

Strongest, cleanest signal in the entire analysis — both variables show a near-monotonic, roughly 2.5-3x increase from lowest to highest band, every point backed by thousands of encounters with the tightest margins seen all session. Stronger and simpler than age, medications, or length of stay individually, and needs no banding caveats or interaction complexity to state to a stakeholder: prior healthcare utilization (inpatient or ER) is the single strongest predictor of 30-day readmission found in this dataset.

**Prior inpatient × prior emergency interaction — the actual #1 finding.** Full 4×4 table, every cell n≥251 (all trustworthy, no exclusions needed):

| prior_inpatient | prior_emergency | rate | n | margin_of_error |
|---|---|---|---|---|
| 0 | 0 | 8.5% | 61,741 | ±0.22 |
| 0 | 3+ | 12.2% | 336 | ±3.50 |
| 3+ | 0 | 24.1% | 4,602 | ±1.24 |
| **3+** | **2** | **34.5%** | **467** | **±4.31** |
| 3+ | 3+ | 34.0% | 739 | ±3.41 |

This is genuine compounding, not redundancy: at `0 prior inpatient`, the emergency-visit range only moves the rate +3.75pp (8.5%→12.2%). At `3+ prior inpatient`, the same emergency range moves it +10.3pp (24.1%→34.5%) — nearly 3x the swing. The combined top cells (33-34%) exceed what either variable's own maximum would predict alone (26.4% inpatient-only, 25.3% emergency-only), confirming the two risks genuinely stack. **~3x the overall 11.4% baseline.**

Minor wobble, not a real reversal: `2 prior inpatient × 2 prior emergency` dips to 15.99% before `3+` climbs to 20.72% — both cells have wider margins (n≈250-300, ±4-5pp), read as noise.

**This is the flagship finding for the write-up:** patients with both 3+ prior inpatient stays and 2+ prior ER visits in the past year show close to a 1-in-3 readmission rate — the single highest-priority group in the dataset for discharge planning resources. Stronger, cleaner, and more reliable than the age×medications or stay×medications interactions found earlier.

---

### Methodology note — prior inpatient × prior emergency interaction (for the end report)

**What was calculated.** Using the same `readmit_summary()` helper as the earlier interactions, readmission rate, group size (`n`), and margin of error were computed for every combination of banded prior inpatient visits (0, 1, 2, 3+) and banded prior emergency visits (0, 1, 2, 3+) in the year before the encounter, to test whether the two utilization measures compound each other or are largely redundant.

**Why both were banded.** `number_inpatient` and `number_emergency` are both heavily right-skewed count variables — 66.7% of encounters have zero prior inpatient visits, and 88.8% have zero prior emergency visits, with long, thin tails beyond that (down to isolated encounters with 20+ prior visits in a single year). Both were banded into 0 / 1 / 2 / 3+ before analysis, partly to keep every group well-populated, and partly because "any prior admission at all" is itself a clinically meaningful distinction, separate from the exact count.

**The formula.** Same as the previous two methodology notes: `rate` = (encounters readmitted `<30`) ÷ (total encounters in the cell) × 100; `margin_of_error` = 1.96 × sqrt(rate×(1−rate)/n) × 100, the standard 95% Wald interval for a proportion.

**Why this was done — testing for compounding vs. redundancy.** Two variables that both measure "how much has this patient used the healthcare system before" could relate to readmission risk in two different ways. If they mostly capture the same underlying signal, a patient already high on one measure gains little additional risk information from also being high on the other — the combined rate would sit close to whichever single variable's own rate is higher. If instead the two capture genuinely different risk information, being high on both should push the rate meaningfully above what either variable predicts alone. Crossing the two banded variables and comparing the combined cells against each variable's individual maximum is a direct way to distinguish these two possibilities.

**What the plot depicts.** A line chart with `prior_inpatient_band` on the x-axis and readmission rate (%) on the y-axis, one colored line per `prior_emergency_band` category, restricted to combinations with `n ≥ 30` (in this case every one of the 16 possible cells cleared that threshold comfortably, with the smallest at n=251). Each line traces how the readmission rate changes across prior-inpatient bands for a fixed level of prior emergency visits — a widening gap between lines at higher inpatient bands, rather than lines staying a constant distance apart, is direct visual evidence of interaction rather than two independent, additive effects.

**What it showed.** At `0` prior inpatient visits, moving across the full range of prior emergency bands shifts the rate by only about 3.75 percentage points (8.5% → 12.2%). At `3+` prior inpatient visits, the same emergency-band range shifts the rate by roughly 10.3 percentage points (24.1% → 34.5%) — nearly three times the swing. The combined highest cells (33-34%, at `3+` inpatient crossed with `2` or `3+` emergency visits) exceed the individual maximum of either variable taken alone (26.4% for inpatient, 25.3% for emergency), which would not be expected if the two variables were simply redundant measures of the same underlying risk. This is evidence of a genuine interaction: prior inpatient and prior emergency utilization compound each other rather than substituting for one another, and patients high on both measures represent a meaningfully higher-risk group than either measure would identify in isolation — a rate close to 1 in 3, roughly three times the dataset's overall 11.4% baseline.

**A caveat noted, not treated as a finding.** The `2 prior inpatient × 2 prior emergency` cell (15.99%) dips slightly below its neighboring `1`-emergency cell (18.25%) before `3+` emergency climbs again to 20.72%. Both of these cells have comparatively wider margins of error (n≈250-300, ±4-5 percentage points), so this small dip is more consistent with sampling noise than a genuine reversal of the overall pattern, and was not treated as informative.

---

## Phase 3 — Part 2: Analytical SQL Queries (CTEs + window functions)

Not a re-run of the Python EDA — only queries that deliberately recreate a Python finding (as a cross-check, or because it's the flagship result worth having live in the database) are labeled as such. The rest are new analysis or exist to demonstrate a SQL-specific technique (window functions) with no direct Python equivalent.

### Query 1 — Readmission rate by diagnosis category × age band (deliberate cross-check)

Recreates the Python finding that resolved the `[20-30)` age spike from Phase 2, using a CTE (`WITH ... AS (...)`) to aggregate first, then filter on `n ≥ 30`. Required a `JOIN` to `dim_diagnosis` to turn `diagnosis_key` back into a readable category — unlike the Python `df`, which already had `diagnosis_category` as a plain column.

**Result: exact match with Python.** `Diabetes × [20-30)` returned `n=677`, `readmit_rate=20.24%` — identical to the Python `readmit_summary()` output. Confirms both implementations are computing the same thing correctly, and gives the SQL side the same trustworthy grounding the Python analysis already had.

**A real SQL-specific bug hit along the way:** `SUM()` does not accept a `BIT` column directly (`readmit_30` is `BIT`) — SQL Server requires an explicit `CAST(readmit_30 AS INT)` first. Unlike Python, where `True`/`False` sums without any special handling, T-SQL treats `BIT` as too restrictive a type for aggregate functions. Also needed an outer `CAST(... AS FLOAT)` around the rate calculation, since SQL Server performs integer division by default and would otherwise silently round every rate to 0 or 1.

### Query 2 — Average length of stay by diagnosis category

**What was calculated.** `AVG(time_in_hospital)` per diagnosis category, using `HAVING COUNT(*) >= 30` instead of a CTE — since the filter here applies directly to an aggregate computed in the same `GROUP BY`, `HAVING` is the more direct tool (`WHERE` can't reference `COUNT(*)`, because `WHERE` filters rows before grouping happens; `HAVING` filters groups after aggregation).

**Why this was done.** Not a Python re-run — length of stay was only examined on its own and interacted with medication count in Phase 2, never broken down by diagnosis category. This is new descriptive information relevant to the hospital-operations framing of the project (bed capacity / resource planning by diagnosis type), separate from the readmission-risk questions the rest of the project focuses on.

**What it showed:**

| diagnosis_category | n | avg_length_of_stay |
|---|---|---|
| Neoplasms | 3,131 | 5.28 |
| Other | 17,793 | 4.77 |
| Injury | 6,853 | 4.62 |
| Digestive | 9,333 | 4.36 |
| Diabetes | 8,661 | 4.33 |
| Genitourinary | 5,002 | 4.22 |
| Circulatory | 29,681 | 4.21 |
| Respiratory | 13,934 | 4.19 |
| Musculoskeletal | 4,935 | 3.91 |

Neoplasms (cancer diagnoses) has the longest average stay at 5.28 days, Musculoskeletal the shortest at 3.91 — a modest ~1.4 day spread, notably smaller than the effect sizes found in the readmission-rate work. All groups are well-powered (smallest n=3,131), so no reliability caveat is needed here, unlike several of the Phase 2 findings. The result is clinically plausible on its face: cancer care typically involves more complex treatment (surgery, monitoring, complications) than the other categories present in this dataset, which is a reasonable real-world explanation rather than a surprising or suspicious pattern.

**Limitation noted.** Unlike the binary readmission-rate work, no margin-of-error / confidence interval was computed for these averages — the Wald formula used throughout this project is specific to proportions (rates), not means. If this needs the same rigor applied later, the standard error of a mean (`stddev / sqrt(n)`) would be the equivalent approach.

### Query 3 — Prior inpatient × prior emergency interaction (deliberate cross-check, flagship finding)

**What was calculated.** `number_inpatient` and `number_emergency` banded into `0`/`1`/`2`/`3+` using `CASE WHEN` — SQL's equivalent of the Python `pd.cut(bins=[-1,0,1,2,100], labels=['0','1','2','3+'])` — then crossed and aggregated the same way as query 1, filtered to `n ≥ 30` via `HAVING`.

**Why this was done.** This is the single strongest finding in the entire project (Phase 2), so it's the one most worth having live in the database rather than only existing in a notebook — the actual candidate for a risk-flag view feeding the Phase 5 dashboard.

**What it showed — exact match with Python, all 16 cells:**

| prior_inpatient | prior_emergency | n | readmit_rate |
|---|---|---|---|
| 3+ | 2 | 467 | 34.48% |
| 3+ | 3+ | 739 | 33.96% |
| 3+ | 1 | 1,006 | 27.44% |
| 3+ | 0 | 4,602 | 24.14% |
| ... | ... | ... | ... |
| 0 | 0 | 61,741 | 8.45% |

Every cell matches the Python `readmit_summary()` output exactly — same `n`, same rate to two decimal places. Confirms the SQL banding logic (`CASE WHEN`) reproduces the Python banding logic (`pd.cut`) correctly, and gives this flagship finding the same database-native, reusable status as query 1's result.

### Query 4 — Ranking diagnosis categories by readmission rate (new: window function)

**What was calculated.** `RANK() OVER (ORDER BY readmit_rate DESC, n DESC)` applied to the per-category readmission rates from query 1's underlying aggregation, with `n DESC` as an explicit tie-break (a larger, more reliable sample wins any tie). First real use of a window function in this project — unlike `GROUP BY`, which collapses rows into one per group, a window function (`OVER(...)`) computes a value across a set of rows without collapsing anything, adding a new column while every row stays visible.

**A reliability bug caught before it became a wrong finding.** The first version of this query had no `HAVING` filter, and ranked `Missing` (encounters with no primary diagnosis recorded) as the #1 riskiest category at 25% — but `n=20` for that group, below the project's n≥30 floor used everywhere else, with a margin of error around ±19 percentage points (true range roughly 6-44%). Fixed by adding `HAVING COUNT(*) >= 30` inside the CTE, excluding unreliable groups from the ranking entirely rather than just visually flagging them after the fact.

**What it showed (reliable categories only):**

| rank | diagnosis_category | n | readmit_rate |
|---|---|---|---|
| 1 | Diabetes | 8,661 | 13.10% |
| 2 | Injury | 6,853 | 12.40% |
| 3 | Circulatory | 29,681 | 11.69% |
| 4 | Other | 17,793 | 11.68% |
| 5 | Genitourinary | 5,002 | 11.04% |
| 6 | Neoplasms | 3,131 | 10.89% |
| 7 | Digestive | 9,333 | 10.82% |
| 8 | Respiratory | 13,934 | 10.06% |
| 9 | Musculoskeletal | 4,935 | 9.54% |

Diabetes ranks highest, consistent with the earlier `[20-30)` age-spike finding — diabetes-primary diagnoses were already shown to carry elevated risk, particularly in younger adults, and this category-level ranking reinforces that same signal at a coarser level. The overall spread is modest (9.54% to 13.10%, ~3.5 percentage points) — much narrower than the interaction findings from query 3, so this ranking is informative but not a dramatic differentiator on its own.

### Query 5 — Gap to next-higher category and to the overall baseline (new: window function)

**What was calculated.** `LAG(readmit_rate) OVER (ORDER BY readmit_rate DESC)` pulls the previous row's rate (the next-higher category) into the current row, without needing a self-join. Two derived columns: `gap_from_next_higher` (how far below the category ranked just above it) and `gap_from_overall_baseline` (`readmit_rate − 11.4`, the dataset-wide rate established back in Phase 2, Step 2).

**Why this was done.** Complements query 4's ranking with a different question — not just "who's highest," but "is the drop-off between categories smooth or does it have a cliff," and "which categories actually sit above the population average, not just above each other."

**What it showed.** `Diabetes` correctly returns `NULL` for both `LAG()`-derived columns (expected — no row ranks above the top one). The category-to-category drop-off is smooth throughout, no cliff: the largest single-step gap is only ~0.76pp (Digestive → Respiratory), and Circulatory → Other is nearly tied (~0.003pp gap).

The baseline comparison is the more actionable framing: exactly 4 of the 9 reliable categories sit *above* the dataset's overall 11.4% rate — Diabetes (+1.70pp), Injury (+1.00pp), Circulatory (+0.29pp), Other (+0.28pp) — while the remaining 5 sit below it, down to Musculoskeletal at -1.86pp. This is a cleaner way to flag which diagnosis groups warrant above-average discharge-planning attention than the ranking alone provides, since it's anchored to a fixed reference point rather than each category's relative position to its neighbors.

**Phase 3, Part 2 complete.** All 5 analytical queries written: 2 deliberate cross-checks against Python (queries 1 and 3, both exact matches), 1 new descriptive finding (query 2, length of stay by diagnosis), and 2 new SQL-only window-function techniques (queries 4 and 5, `RANK()` and `LAG()`). Next: Part 3, building 3 reusable SQL views for Power BI DirectQuery consumption in Phase 5.

---

## Phase 3 — Part 3: Reusable SQL Views (`readmit_summary()`, natively in SQL)

**What was calculated.** Three `CREATE VIEW` objects, each reproducing the Python `readmit_summary()` pattern (`rate`, `n`, `margin_of_error`) directly in T-SQL, grouped by a different dimension: `v_readmit_summary_by_diagnosis`, `v_readmit_summary_by_age`, `v_readmit_summary_by_utilization` (prior inpatient × prior emergency, banded via the same `CASE WHEN` logic as query 3). The margin-of-error formula uses `SQRT()` — the same Wald interval used throughout the entire project: `1.96 × SQRT(rate × (1-rate) / n) × 100`.

**Why a view, not just a query.** Unlike every query run so far in Phase 3, a view is a saved, permanent, named object in the database — it persists after the query window closes and can be queried like a table by anything with database access, including Power BI's DirectQuery in Phase 5. This is the actual deliverable that replaces a one-off script's output with a reusable analytical asset — the core reason Phase 3 was redirected away from a repeat star-schema build in the first place.

**A deliberate design choice: no `HAVING n >= 30` baked into the views.** Every other analysis in this project applied that reliability floor directly. These views don't — `n` and `margin_of_error` are exposed as columns precisely so any downstream consumer (Power BI, another analyst) can apply their own threshold at query time, rather than being locked into a filter decided months earlier. Filtering happens with `WHERE n >= 30` on top of the view when needed, same as it would on a real table.

**Verification — all three matched known values exactly:**
- `v_readmit_summary_by_diagnosis`: `Missing` (n=20) returned a margin of error of ±18.98 — matches the ~19pp estimate that flagged it as unreliable back in query 4. `Circulatory` (n=29,681, largest group) showed the tightest margin (±0.37); `Neoplasms` (n=3,131, smallest reliable group) the widest (±1.09) — margin shrinking as n grows, as the formula predicts.
- `v_readmit_summary_by_age`: exact match with the Phase 2 age-band table across all 10 bands, including `n` — e.g. `[20-30)` at 14.32% (n=1,649).
- `v_readmit_summary_by_utilization`: exact match with query 3 across all 16 cells, now with margin of error added — `3+ inpatient × 2 emergency` at 34.48% ± 4.31 (n=467), `0×0` baseline at 8.45% ± 0.22 (n=61,741).

**Phase 3 complete.** Part 1 (lean `fact_encounter` + `dim_diagnosis` schema), Part 2 (5 analytical queries — 2 cross-checks, 1 new finding, 2 window-function techniques), Part 3 (3 reusable views, ready for Phase 5 DirectQuery). Next: Phase 4, the Python predictive layer (logistic regression).

---

## Phase 4 — Predictive Layer (Logistic Regression)

**Unit of analysis.** Restricted to first encounter per patient (69,990 rows, down from the 99,343-encounter working set), matching the source paper's approach — logistic regression assumes independent observations, and repeat encounters from the same patient violate that.

**Features.** Exactly the 7 candidates Phase 2's EDA validated (`02_EDA_Notes.md`, Step 3): `number_inpatient`, `number_emergency`, `time_in_hospital`, `num_medications` (kept continuous, not banded — the model fits its own slope), plus `age`, `medical_specialty`, `diagnosis_category` (one-hot encoded). `age` deliberately kept categorical rather than converted to a single numeric value, since Phase 2 found a non-monotonic `[20-30)` spike a linear age term would misrepresent.

**Model.** `LogisticRegression(class_weight='balanced')` — balanced weighting needed because of the 11.4% base rate; without it the model tends to just predict "not readmitted" for nearly everyone. 80/20 train/test split, stratified on the target.

**Baseline performance:** AUC = 0.607, recall (class 1) = 0.55, precision (class 1) = 0.12. Modest — better than random, far from strong. AUC measures ranking ability across the whole population: the probability a randomly chosen readmitted patient scores higher than a randomly chosen non-readmitted one.

**Tested an explicit `number_inpatient × number_emergency` interaction term** (recreating the flagship Phase 2/SQL-query-3 finding directly as a model feature) — AUC barely moved (0.6074 → 0.6073), no real change. Reason: AUC aggregates ranking ability across the *entire* test set, but the interaction only meaningfully affects a small high-utilization subgroup (~1,200 of 69,990 patients have 3+ prior inpatient and 2+ prior emergency visits). A pattern that's genuinely large within a small subgroup can have negligible leverage on a whole-population ranking metric. Dropped the interaction term — added complexity without benefit. Practical implication for the write-up: this finding is still real and actionable, just better suited to an explicit rule-based risk flag (same idea as the LACE index) than to moving an aggregate model metric.

**Coefficient reliability check — caught the same small-n trap as SQL query 4.** Sorting all coefficients by odds ratio, 7 of the 20 most extreme values (both directions) came from `medical_specialty` categories with fewer than 30 patients (e.g. `Pathology` n=6, `Anesthesiology` n=7, `Psychiatry-Child/Adolescent` n=6) — the same reliability floor used everywhere else in this project applies here too; a coefficient from a near-empty category is not a trustworthy finding, regardless of how large it looks.

**Reliability filter, done properly (not hand-picked).** One-hot encoding loses the link back to each category's sample size, so a `get_n()` lookup was built to reattach `n` to every coefficient (matching each `medical_specialty_X` / `diagnosis_category_X` / `age_X` feature name back to its original `value_counts()`), then filtered to `n ≥ 30` — same reliability floor as every other analysis in this project, applied programmatically instead of eyeballed.

**Reliable risk-increasing findings (n≥30), top 10 by odds ratio:**

| feature | n | odds_ratio |
|---|---|---|
| medical_specialty_Hematology/Oncology | 109 | 3.76 |
| medical_specialty_Oncology | 205 | 2.73 |
| medical_specialty_Surgeon | 40 | 1.96 |
| medical_specialty_Psychiatry | 613 | 1.82 |
| medical_specialty_Gastroenterology | 384 | 1.78 |
| medical_specialty_PhysicalMedicineandRehabilitation | 194 | 1.78 |
| medical_specialty_Nephrology | 797 | 1.78 |
| medical_specialty_Surgery-Vascular | 359 | 1.74 |
| medical_specialty_Orthopedics | 1,128 | 1.68 |
| age_[80-90) | 11,110 | 1.65 |

`age_[80-90)` is the first non-specialty feature to appear, and it's backed by a very large sample (n=11,110) — a genuinely strong, trustworthy age effect, consistent with Phase 2's finding that the 70+ bands plateau at an elevated rate.

**Reliable risk-decreasing findings (n≥30), bottom 10 by odds ratio:**

| feature | n | odds_ratio |
|---|---|---|
| diagnosis_category_Digestive | 6,488 | 0.77 |
| medical_specialty_Surgery-Cardiovascular | 85 | 0.77 |
| medical_specialty_Endocrinology | 97 | 0.73 |
| diagnosis_category_Respiratory | 9,491 | 0.69 |
| medical_specialty_Hospitalist | 36 | 0.59 |
| medical_specialty_Ophthalmology | 35 | 0.55 |
| medical_specialty_Otolaryngology | 110 | 0.47 |
| medical_specialty_Pediatrics-CriticalCare | 73 | 0.44 |
| medical_specialty_Gynecology | 54 | 0.13 |
| medical_specialty_Pediatrics-Endocrinology | 147 | 0.07 |

`Pediatrics-Endocrinology` (n=147) shows the largest risk reduction found anywhere in the model, and it's clinically plausible — a distinct, likely younger and less comorbid population. `diagnosis_category_Digestive`/`Respiratory` have very large sample sizes (6,488 / 9,491) but need one interpretation caveat: `drop_first=True` dropped `Circulatory` (alphabetically first) as the implicit reference category, so these two are shown as lower-risk *relative to Circulatory specifically*, not in absolute terms — consistent with Phase 3's `RANK()` query, where Circulatory placed 3rd highest overall.

**The 4 continuous features — the most defensible findings, no small-n caveat needed (n=69,990 each):**

| feature | odds_ratio | interpretation |
|---|---|---|
| number_inpatient | 1.43 | each additional prior inpatient visit: +43% odds |
| number_emergency | 1.15 | each additional prior ER visit: +15% odds |
| time_in_hospital | 1.05 | each additional day in hospital: +5% odds |
| num_medications | 1.007 | each additional medication: +0.7% odds (compounds over range) |

**Cross-validation with Phase 2:** direction and relative ranking match exactly — all four increase risk, ordered `number_inpatient` > `number_emergency` > `time_in_hospital` > `num_medications`, identical to Phase 2's percentage-point effect-size ranking (17.8pp > 14.6pp > 6.4pp > 5.4pp). The univariate EDA and the multivariate model tell the same story, a real confirmation point for the write-up.

**Risk score export.** The fitted model was trained on the first-encounter-only subset (69,990 rows, required for valid model fitting), but scored against the *full* 99,343-encounter working set — a hospital dashboard needs a risk score for every current encounter, not just first-time patients. Used `predict_proba()` rather than the hard 0/1 `predict()` output, and exported the raw probability (`encounter_risk_scores.csv`, `encounter_id` + `readmit_risk_score`), not a pre-baked high/low binary flag — same philosophy as the SQL views not hardcoding an `n ≥ 30` cutoff. Power BI defines its own "high risk" threshold visually in Phase 5, rather than that decision being locked in here.

**Calibration caveat, documented not hidden.** `class_weight='balanced'` re-weighted the loss function during training, so these probabilities are not necessarily calibrated — a score of 0.30 doesn't reliably mean "30% of similar patients get readmitted." They're valid for *ranking* patients relative to each other (exactly what a risk-flag dashboard needs), not for reading as literal probabilities.

**Visual sanity check — three plots, all confirming the numbers, no red flags.**
- **ROC curve:** bows modestly above the diagonal throughout, consistent with AUC = 0.607 — visibly better than random, clearly not strong separation.
- **Predicted risk score distribution, by actual outcome:** the two groups overlap heavily (both clustered ~0.25-0.85), but `Readmitted` is visibly shifted right of `Not readmitted` — more mass in the 0.55-0.85 range, less in 0.25-0.45. Exactly what a modest-but-real signal looks like: a genuine shift, not a clean split. Also visible: neither distribution approaches 0 or 1 — predictions stay clustered toward the middle, a direct visual confirmation of the calibration caveat above (`class_weight='balanced'` pushes the model away from confident, extreme predictions).
- **Confusion matrix heatmap:** makes the precision problem visually obvious — the "Not readmitted → Not readmitted" cell (7,766) dominates the color scale, while "Readmitted → Readmitted" (688 true positives) is small and pale next to a much larger false-positive cell (4,975).

**Phase 4 complete.** Model trained and evaluated (AUC 0.607, baseline honestly reported, interaction term tested and dropped after showing no benefit), coefficients extracted and reliability-filtered (n≥30), cross-validated against Phase 2's findings, risk scores exported for all 99,343 encounters, results visually sanity-checked. Next: Phase 5, Power BI dashboard via DirectQuery to Phase 3's SQL views, plus this risk score export.

---

## Phase 5 — Power BI Dashboard

**Risk scores loaded into Azure SQL.** `risk_scores` table (`encounter_id` PK/FK to `fact_encounter`, `readmit_risk_score` FLOAT) created and loaded via the same staging-table pattern established in Phase 3, Part 1 — keeps the whole pipeline (raw data, dimension table, analytical views, model output) living in one place for Power BI to connect to, rather than mixing a CSV import alongside a database connection. Verified two ways: `SELECT COUNT(*) FROM risk_scores` = 99,343, and a `LEFT JOIN` from `fact_encounter` confirms 0 encounters missing a score — complete, consistent load.

**Model relationships fixed in Power BI's data model.** `risk_scores ↔ fact_encounter` needed its cardinality manually corrected from Power BI's auto-detected many-to-one to the actually-correct one-to-one (both `encounter_id` columns are unique primary keys). `dim_diagnosis ↔ fact_encounter` was auto-detected correctly as one-to-many and left as-is.

**KPI measures, and a real DAX gotcha.** Three measures built: `Readmission Rate`, `Avg Length of Stay`, `% High Risk`. Hit the same "can't SUM a boolean" issue DAX-side that T-SQL hit back in Phase 3, Part 2 — `readmit_30` is `BIT` in SQL Server, which Power BI maps to its own Boolean type, and DAX's `SUM()` rejects it the same way `SUM()` did in T-SQL. Fixed by rewriting as `CALCULATE(COUNTROWS(...), condition = TRUE())` instead of summing, mirroring the `% High Risk` measure's existing pattern. Also hit a formatting trap: DAX measures already multiplying by `* 100` (matching this project's Python/SQL convention of expressing rates as "11.39" not "0.1139") conflict with Power BI's built-in Percentage format, which expects a raw 0-1 decimal and auto-multiplies — applying both would double-scale the display. Fixed by dropping `* 100` from the DAX and using Power BI's built-in `%` format button instead, the more idiomatic approach in this tool specifically.

**Diagnosis breakdown chart — caught the `Missing` (n=20) issue a third time.** Built a bar chart directly from `v_readmit_summary_by_diagnosis`. Two things needed fixing before it was trustworthy: the aggregation defaulted to "Sum of readmit_rate" (wrong — rates aren't additive, should always be Average/Min/Max, never Sum, even though the numbers looked identical here since the view already has one row per category), and `Missing` (n=20) appeared as the visually top-ranked bar with no indication it's unreliable. Fixed with a visual-level filter (`n ≥ 30`) on the chart specifically — same reliability floor caught in SQL query 4 and the Phase 4 model coefficients, now caught a third time in the dashboard itself. The view intentionally doesn't bake this filter in (so any consumer can choose their own threshold), which means each *consumer* — this dashboard included — has to actually apply it, not assume it's already handled.

**A real finding: the `% High Risk` KPI (53.21%) is misleading on its own.** The risk model (Phase 4) was trained and evaluated only on first-encounter-only data (69,990 rows) — but the dashboard's `% High Risk` measure scores the *full* 99,343-encounter population, including repeat encounters that were never part of training or testing.

Checked directly: `df.groupby('is_first_encounter')['readmit_risk_score'].mean()` and the `% flagged high-risk` broken out by group —

| group | % flagged high-risk (≥0.5) |
|---|---|
| Repeat encounter | 82.93% |
| First encounter | 40.74% |

More than double. The dashboard's 53.21% is a weighted blend of the two: repeat encounters are 29,353 of 99,343 (29.6%), first encounters are 69,990 (70.4%) — `0.296 × 82.93 + 0.704 × 40.74 ≈ 53.2%`, matching the dashboard figure almost exactly, confirming this is the actual mechanism, not a coincidence.

**Why this happens.** Repeat encounters aren't a random sample — by the time a patient has a second or third recorded encounter, they've had more opportunity to accumulate prior inpatient/emergency visits than they had at their very first encounter. Since `number_inpatient` (OR=1.43) and `number_emergency` (OR=1.15) are the two strongest risk-increasing features in the model, repeat encounters get pushed toward high-risk scores far more often — a structural artifact of using a first-encounter-only model to score a mixed population, not a sign the model is broken.

**Resolution:** the 53.21% figure isn't wrong, but reporting it alone overstates population-wide risk. Decided to split the dashboard's single `% High Risk` KPI into two segments (first-encounter vs. repeat-encounter) rather than a single blended number, so the dashboard doesn't imply a homogeneity in risk that isn't there.

**Implemented.** Added `is_first_encounter` (BIT) directly to `fact_encounter` in Azure SQL — computed via `ROW_NUMBER() OVER (PARTITION BY patient_nbr ORDER BY encounter_id)`, the same window-function pattern from Phase 3, and the same "first encounter" definition already used in the Python model. (Hit a classic SQL Server gotcha along the way: `ALTER TABLE ADD COLUMN` and a statement referencing that new column can't run in the same batch — SQL Server compiles the whole batch before executing any of it, so the new column doesn't exist yet at compile time. Fixed with `GO` to force two separate batches.) Verified via `SUM(CAST(is_first_encounter AS INT))` = 69,990, matching exactly.

Replaced the single `% High Risk` DAX measure with two — `% High Risk (First Encounter)` and `% High Risk (Repeat Encounter)` — each using an outer `CALCULATE` filtering on `fact_encounter[is_first_encounter]`, relying on the `risk_scores ↔ fact_encounter` one-to-one relationship (with "Both" cross-filter direction) to propagate the filter correctly. Confirmed in the dashboard: 40.74% / 82.93%, an exact match to the Python-side check. Closes this finding out — the dashboard no longer reports a single blended, misleading figure.

**Admission source breakdown — a 4th view, added mid-Phase-5 (exploratory, unlike the other three).** Unlike diagnosis category and age, `admission_source_id` was never validated in Phase 2's EDA — a quick raw SQL check first (`GROUP BY admission_source_id HAVING n >= 30`) confirmed a real, meaningful spread (9.47%-15.68%) before investing further effort.

Built `dim_admission_source` (25 rows) by parsing directly from `IDS_mapping.csv`'s third stacked table (`skiprows=42`) rather than retyping labels by hand — hit the exact same bug class as Phase 1's `'?'` issue: pandas' default missing-value list includes the literal string `"NULL"`, which silently converted `admission_source_id=17`'s real description (`"NULL"`) into an actual `NaN`, then `.dropna()` removed that row entirely (25 rows became 24). Fixed with `keep_default_na=False`, same fix as Phase 1.

Loaded via the established CSV/staging pattern, plus a genuinely new SQL concept: `ALTER TABLE ADD COLUMN` and any statement referencing that new column can't run in the same batch (SQL Server compiles the whole batch before executing any of it) — required a `GO` separator, first hit while adding `is_first_encounter` earlier this phase, confirmed again here.

**A second kind of unreliability, distinct from small-n.** Three of the ten raw `admission_source_id` codes with `n ≥ 30` (`Not Available`, `NULL`, `Not Mapped`) aren't real admission sources — they're placeholder codes for missing data. Statistically fine (decent sample sizes), but semantically meaningless as individual dashboard categories. Collapsed into a single `Unknown / Not Recorded` bucket in the view itself (`CASE WHEN`), verified the bucketed `n` (6,854) exactly matches the sum of the three raw codes' counts (159 + 6,570 + 125). In the dashboard chart, recolored that bar separately (maroon vs. blue) rather than hiding it — excluding it would make the categories add up to less than the full population with no explanation, but leaving it visually identical to real findings would be misleading the other way.

**Also re-surfaced a documented formula limitation.** The un-filtered raw view showed several very-small-n admission sources (`n` of 1-8) with `readmit_rate = 0` and `margin_of_error = 0` — the same Wald-formula blind spot documented back in Phase 2, Step 4 (the formula always returns exactly 0 at an observed rate of 0% or 100%, regardless of n). Confirms the `n ≥ 30` filter is doing real work here, not just following convention.

**Risk-flag visual — a watchlist table, not another aggregate chart.** The KPI cards and breakdown charts already cover population-level summaries; this visual is deliberately different — a sortable table of the highest-risk *current* encounters (`encounter_id`, `age`, `diagnosis_category`, `number_inpatient`, `number_emergency`, `readmit_risk_score`), the genuinely actionable piece for a hospital-ops audience ("who to prioritize today," not just "what's the overall rate").

Two aggregation bugs caught building it, same root cause both times: Power BI defaults numeric fields to some aggregation even in a one-row-per-encounter detail table. Table columns (`number_inpatient`, `number_emergency`, `readmit_risk_score`) needed **"Don't summarize"** instead of Sum/Count — `Count of readmit_risk_score` was showing `1` for every row (a count of one value) instead of the actual score. Separately, the **Top N filter's "By value" field** had the identical bug — it defaulted to `Count of readmit_risk_score`, meaningless since every row's count is `1`, so the filter had no real basis to rank rows and grabbed an arbitrary (in this case, lowest-risk) 50 rows instead of the highest. Fixed by changing the filter's "By value" aggregation to **Sum** — correct here since each row is a unique encounter, so summing one value just returns that value.

**Confirmed correct:** the final table shows scores clustering at 0.99-1.00, with heavy prior inpatient counts (11-17) and high prior emergency visits (up to 20-28) — directly consistent with `number_inpatient`/`number_emergency` being the two strongest risk-increasing features in Phase 4's model.

**Phase 5 dashboard build complete.** 3 KPI cards (30-Day Readmission Rate, Avg. Length of Stay, and the split % High Risk by first/repeat encounter), 3 breakdown charts (diagnosis category, age, admission source — the last one newly built mid-phase, including a new `dim_admission_source` table and 4th SQL view), and the risk-flag watchlist table. All connected via DirectQuery, all filtered to the project's `n ≥ 30` reliability standard where applicable. Next: Phase 6, the stakeholder findings write-up.

---

## Phase 3 — Step 1: ICD-9 Diagnosis Categorization

**What was calculated.** `diag_1` (the primary diagnosis, stored as a raw ICD-9 code — e.g. `"428.0"`, `"250.83"`, `"V27"`) was mapped to one of 9 readable clinical categories via a `categorize_diagnosis()` function, producing a new `diagnosis_category` column. The function checks each code in order: missing → `"Missing"`; starts with `V` or `E` → `"Other"`; otherwise, convert to a number and match it against the ranges below.

**The rule — two principles stacked together, not invented from scratch:**

**1. ICD-9's own chapter structure.** ICD-9 (the WHO/CDC diagnosis coding standard used in US hospital billing) isn't numbered randomly — codes are pre-organized into "chapters" by body system, the same code range means the same disease category at every hospital that uses it. This is the coding standard's built-in structure, not a scheme built for this project:

| Category | ICD-9 range | % of encounters (Strack et al.) |
|---|---|---|
| Circulatory | 390–459, 785 | 30.6% |
| Respiratory | 460–519, 786 | 13.6% |
| Digestive | 520–579, 787 | 9.3% |
| Diabetes | 250.xx | 8.2% |
| Injury | 800–999 | 6.7% |
| Musculoskeletal | 710–739 | 5.8% |
| Genitourinary | 580–629, 788 | 4.9% |
| Neoplasms | 140–239 | 3.6% |
| Other | everything else (mental, skin, blood, infectious, pregnancy, congenital, symptoms/ill-defined, injury-external-cause `V`/`E` codes) | 17.3% combined |

**2. A frequency cutoff on top of the chapter structure.** ICD-9 has ~17 official chapters, but Strack et al. only kept the ones common enough in *this diabetic population* to analyze reliably — any chapter under 3.5% of encounters was folded into `"Other"`, since a category that thin can't support a trustworthy rate (same n-and-margin-of-error logic as every groupby in this project). Diabetes (250.xx) is kept as its own category by exception, despite being one of the smaller chapters by volume, because it's the clinical focus of the whole dataset.

**`V`/`E` code exception.** ICD-9 has a separate alphanumeric sub-system for supplementary classification (`V` codes — e.g. factors influencing health status) and external causes of injury (`E` codes). These fall outside the numeric chapter ranges entirely, so the function checks for them first and routes them straight to `"Other"` rather than attempting a numeric conversion.

**Why this was done.** `diag_1` has 848 distinct raw codes — too granular to chart or trust a rate on individually (most codes have too few encounters). Collapsing to 9 categories makes "does readmission risk vary by diagnosis type?" answerable, and it's the piece needed to finally resolve the open `[20-30)` age-spike question from Step 4 (Phase 2) — whether that spike is linked to a specific diagnosis mix rather than medication load (already ruled out via the age × med_count_band interaction). Using Strack et al.'s exact published scheme, instead of inventing new buckets, also means the categories are directly comparable to a peer-reviewed analysis — citable in the final report.

**What it feeds.** This category becomes `dim_diagnosis` in the Phase 3 SQL schema (`fact_encounter` + `dim_diagnosis`), and the diagnosis-category × age-band query is the first of the 5 planned analytical SQL queries.

**Verified against the source paper** (`df['diagnosis_category'].value_counts(normalize=True) * 100`, working set = 99,343):

| Category | This dataset | Strack et al. |
|---|---|---|
| Circulatory | 29.9% | 30.6% |
| Other | 17.9% | 17.3% |
| Respiratory | 14.0% | 13.6% |
| Digestive | 9.4% | 9.3% |
| Diabetes | 8.7% | 8.2% |
| Injury | 6.9% | 6.7% |
| Genitourinary | 5.0% | 4.9% |
| Musculoskeletal | 5.0% | 5.8% |
| Neoplasms | 3.2% | 3.6% |
| Missing | 0.02% | — |

Every category lands within ~1pp of the published figures, and `Missing` (0.02%) matches Step 1's 21 missing `diag_1` values exactly (21 ÷ 99,343 = 0.021%). Confirms `categorize_diagnosis()` is implemented correctly — same verify-against-source-paper check used throughout this project.

---

### Resolved: the `[20-30)` age spike (open question from Phase 2, Step 4)

**What was calculated.** `readmit_summary(df, ['age', 'diagnosis_category'])`, filtered to `n ≥ 30`, first isolated to just the `[20-30)` age band and sorted by rate, then re-filtered to just the `Diabetes` category across every age band. Same `readmit_summary()` helper, same 95% Wald margin-of-error formula as every other groupby in this project — no new statistical method, just a new pair of columns to cross.

**Step 1 — which diagnosis is driving `[20-30)`'s 14.3% rate?**

| diagnosis_category | rate | n | margin_of_error |
|---|---|---|---|
| Diabetes | 20.24% | 677 | ±3.03 |
| Digestive | 18.32% | 131 | ±6.62 |
| Circulatory | 12.00% | 50 | ±9.01 |
| Other | 9.68% | 568 | ±2.43 |
| Injury | 8.47% | 59 | ±7.11 |
| Respiratory | 6.67% | 75 | ±5.65 |
| Genitourinary | 6.56% | 61 | ±6.21 |

Diabetes is the only reliable standout — largest cell (n=677), tightest margin (±3.03), and clearly above both the age band's own 14.3% average and the `Other` category in the same age band (9.68%, n=568, also tight). Digestive (18.32%) looks close but its margin (±6.62) is too wide to trust as a second real finding; Circulatory (n=50, ±9.01) is barely above the reliability floor and shouldn't be cited at all.

**Step 2 — is Diabetes generally risky, or specifically risky at this age?** (the main-effect-vs-interaction test)

| age | rate | n | margin_of_error |
|---|---|---|---|
| 0-10 | 1.48% | 135 | ±2.04 |
| 10-20 | 5.05% | 475 | ±1.97 |
| **20-30** | **20.24%** | **677** | **±3.03** |
| 30-40 | 13.37% | 905 | ±2.22 |
| 40-50 | 13.75% | 1,440 | ±1.78 |
| 50-60 | 12.70% | 1,449 | ±1.71 |
| 60-70 | 12.11% | 1,354 | ±1.74 |
| 70-80 | 14.88% | 1,344 | ±1.90 |
| 80-90 | 11.84% | 760 | ±2.30 |
| 90-100 | 12.30% | 122 | ±5.83 |

**Why this was done.** A diagnosis category could be linked to an age spike two different ways: either that diagnosis is generally higher-risk at every age (a main effect, in which case the age band just happens to contain more of it — a composition effect, not a true interaction), or that diagnosis is specifically more dangerous at that age (a genuine interaction). Isolating `Diabetes` across every age band, instead of just looking at `[20-30)` alone, is what distinguishes the two.

**What it showed.** Every adult band from 30-40 through 90-100 clusters tightly around 12-15% for diabetes-primary encounters — `[20-30)` alone sits at 20.24%, roughly 5-8 percentage points above every neighboring band. The margins confirm this is a real gap, not noise: `[20-30)`'s range (17.2-23.3%) doesn't overlap with `[30-40)`'s (11.2-15.6%) or `[40-50)`'s (12.0-15.5%). This is a genuine age × diagnosis interaction, not a composition artifact — young adults specifically carry elevated readmission risk when diabetes is their primary diagnosis, a pattern not present at any other age.

**Clinical plausibility (for the write-up).** This age range is a recognized "transition gap" in diabetes care — patients moving from pediatric to adult-managed care, with documented higher rates of DKA recurrence and treatment non-adherence during this period. Lends external credibility to a data-driven finding, similar to the LACE index tie-in for the prior-utilization finding.

**Resolves the open question from Phase 2, Step 4.** The `[20-30)` age spike is not explained by medication load (ruled out earlier — the spike appeared across every medication band equally) but is explained by diagnosis mix: this age band has both an unusually high share of diabetes-primary encounters (~41% of the band) and a genuinely elevated readmission rate within that subgroup specifically.

---

## Recreating Strack et al.'s published figures (Python + Power BI cross-check)

**Goal.** Beyond validating individual findings against the source paper, attempted a direct recreation of the paper's own 3 published figures — a stronger cross-check than comparing summary statistics, since it requires reproducing their actual modeling approach, not just their descriptive numbers. Split deliberately: Python for anything requiring model fitting (Power BI/DAX cannot fit a logistic regression with an interaction term), Power BI for the actual chart visuals, so the recreation lives in both places consistent with this project's established division of labor.

**Figure 2 (age vs. logit of readmission rate) — the simple one.** No new modeling needed, just a transform of an existing rate. One correction made before recreating it: the paper's population is first-encounter-only (69,984 patients); the original Phase 2 age table used the full 99,343-encounter working set. Recomputed on the first-encounter-only subset (69,990 rows) to genuinely match their methodology, in both Python (`df_model.groupby('age')['readmit_30'].mean()` + `np.log(p/(1-p))`) and Power BI (a new view, `v_readmit_summary_by_age_first_encounter`, filtered on `is_first_encounter`, plus a DAX logit measure). **Result: matches the paper's own description exactly** — a steep rise through `[0,30)`, a flat plateau through `[30,60)`, and a second rise through `[60,100)`, the same three-segment shape that motivated the paper's own 3-category age split.

**Figures 1 and 3 (adjusted readmission rate by diagnosis × HbA1c) — the hard ones.** These needed a second, separate model from Phase 4's — an *inferential* model (p-values, confidence intervals) rather than Phase 4's *predictive* one, which meant switching tools: `sklearn.LogisticRegression` (Phase 4) doesn't expose p-values or CIs at all; `statsmodels.GLM` (this exercise) does, and its formula API allows naming exact reference categories to match the paper's chosen baselines.

**New features pulled in specifically for this exercise** (none were in `fact_encounter.csv`, since none were part of the original 7-feature EDA shortlist): `A1Cresult` and `change`, recoded into the paper's exact 4-level HbA1c variable (Not measured / Normal / High-changed / High-not-changed). Hit the `"None"`-as-literal-NaN bug a third time here (same family as the `'?'` and `"NULL"` bugs — pandas' default `na_values` list also includes the string `"None"`) — before the fix, every untested patient was silently miscoded as "High" risk instead of "Not measured," which would have badly distorted the model. Fixed with the same `keep_default_na=False` pattern used twice before. Verified against the paper's own quoted statistic — *"less than half of patients (42.5%) had a medication change"* among untested patients — and got **42.7%**, a very close match confirming the recode was correct.

Also pulled and recoded `race`, `discharge_disposition_id`, and `admission_source_id` into the paper's collapsed categories (race: African American/Caucasian/Missing/Other; discharge: Home/Other; admission: Emergency/Referral/Other) to match their Table 4 model structure. One of these recodes needed the same NaN-vs-literal-string fix again — `race`'s `'?'` was already converted to NaN by the `na_values=['?']` load parameter, so checking for the literal string `'?'` in the recode function silently missed it; fixed by checking `pd.isna()` instead. Medical specialty (Cardiology/General practice/Internal medicine/Missing/Other/Surgery) is the one recode **not fully specified by the paper's text** — used a documented, reasonable interpretation (exact match for the three named specialties, anything containing "Surg" grouped as Surgery, everything else folded into Other) rather than claiming an exact replication.

**Model.** `statsmodels.glm()`, binomial family, formula API with explicit `Treatment(reference=...)` for every categorical variable, including a `diagnosis_category * hba1c_category` interaction term (the term that makes the HbA1c effect diagnosis-specific — the paper's central claim). One reliability issue caught before trusting the output: the `Missing` diagnosis category (n=21 in the full dataset) produced an absurd interaction coefficient (-19.3, SE≈2×10⁴) — a near-empty cell causing quasi-complete separation, the same n≥30 trap hit repeatedly elsewhere in this project (SQL query 4, Phase 4 coefficients, the Power BI risk tiers), this time in a fourth tool. Fixed by excluding `Missing`-diagnosis rows from this specific model, same principle as every other reliability filter applied throughout.

**Coefficient cross-check against the paper's Table 4/5 (honest, not cherry-picked).** Several terms replicate closely — `admission_category[Referral]` nearly identical (-0.021 vs. paper's -0.020); the HbA1c main effects match the paper's headline story (both "High" categories significantly *reduce* readmission risk relative to no testing, same direction and similar significance as theirs). The diagnosis × HbA1c interaction for **Circulatory replicated as significant** (p=0.009, p=0.019 for the two "High" levels), directly reproducing the paper's core claim that diabetes patients' HbA1c-readmission relationship differs from circulatory patients' (their own reported P<0.001 for that comparison). Respiratory's individual interaction terms weren't significant here (p=0.86, p=0.16) — though the paper's own "borderline significant" (P=0.02) came from a joint 3-degree-of-freedom test across all three HbA1c terms at once, a stricter/different check than comparing individual terms, so this isn't necessarily a contradiction. **One genuine, unresolved divergence:** the paper found `age <30` sharply *increases* risk (+1.833, p=0.031); this replication shows essentially no effect (-0.056, not significant) — plausibly because collapsing `[0-10)`+`[10-20)`+`[20-30)` into one `<30` bucket (the paper's own grouping) dilutes the `[20-30)`-specific, diabetes-specific spike this project found back in Phase 3. Documented honestly rather than smoothed over.

**Figure 1 result (Diabetes/Respiratory/Circulatory × HbA1c, reference values + mean `time_in_hospital`, 95% CI via `get_prediction()`).** Strong qualitative match to the paper: **Diabetes shows a clear declining pattern** as HbA1c gets tested (6.50%→6.31%→4.92%→3.91% across Not measured/Normal/High-changed/High-not-changed) — testing, regardless of result, associated with lower risk, the paper's central finding. **Circulatory stays essentially flat** (6.03%→5.87%→6.62%→5.86%) — no such protective pattern, matching the paper's claim that circulatory patients don't show it. **Respiratory is mixed** (a dip then partial recovery), consistent with the paper's own "borderline" framing. Absolute rates run somewhat lower than the paper's reported 0.02-0.11 range (this replication clusters ~0.03-0.07) — plausible given differences in exact covariate specification — but the shape, which is the actual finding, replicates.

**Figure 3 result (all 9 diagnosis categories × HbA1c).** Every cell clears the n≥30 floor (smallest: Neoplasms "High, not changed" at n=58), but clearing 30 isn't the same as being equally trustworthy — the same lesson from Phase 2 (*"n≥30 is a rough floor, not a guarantee"*). CI width tracked cell size exactly as expected: Neoplasms/Genitourinary/Musculoskeletal (all with cells in the 58-230 range) had visibly the widest, noisiest confidence intervals. Split into two panels by data quality rather than guessing the paper's own (unspecified) panel assignment: **Panel A** (Diabetes, Circulatory, Respiratory, Digestive, Other — larger cells, tighter CIs) and **Panel B** (Genitourinary, Injury, Musculoskeletal, Neoplasms — smaller cells, wider CIs, flagged as lower-confidence).

**Power BI build notes.** Both prediction tables exported to `hba1c_diagnosis_predictions` (36 rows: 9 diagnoses × 4 HbA1c levels), loaded via Import Flat File directly (new table, no staging-table detour needed this time — that pattern is only required when loading into an *existing* table). Figure 1 rebuilt as 3 line charts (one per diagnosis, filtered), each with `predicted_rate`/`ci_lower`/`ci_upper` as three separate lines styled as a band (solid colored center line, dashed gray bounds) — Power BI's core Line chart has no native error-bar feature in this version, so this is an approximation using only built-in chart types. Figure 3 built as **one small-multiples line chart** faceted by `diagnosis_category`, avoiding a repetitive 9-visual manual build.

**A new DAX gotcha hit while sorting the HbA1c axis: a calculated column can't be its own sort-by ancestor.** Built `hba1c_order` as a DAX calculated column (`SWITCH` on `hba1c_category`), then tried to set `hba1c_category`'s "Sort by column" property to `hba1c_order` — Power BI rejected it with a circular dependency error (`hba1c_category` → `hba1c_order` → `hba1c_category`), even though the actual values would resolve fine outside Power BI's own dependency tracking. Fixed by defining the order column at the **SQL** level instead (either a hand-written `CASE WHEN`, or — for the age-axis version — extracting the number directly from the existing label text via `SUBSTRING`/`CHARINDEX`), so it's a genuinely independent source column rather than one DAX-derived from the column it's meant to sort. General principle: a "sort by column" target must not depend on the column it's sorting, even indirectly through a DAX formula.

---

### Risk-flag visual, revised — scatter plot with risk tiers (replaces the watchlist table read as "too visually heavy")

**What changed.** The Top-50 watchlist table (previous section) was functionally correct but hard to scan — 50 rows of numbers don't reveal a pattern the way a chart does. Replaced with a bubble scatter: `number_inpatient` (x) × `number_emergency` (y), one bubble per encounter group, colored by a new `Risk Tier` category (Low/Medium/High) and sized by average `readmit_risk_score`.

**Why color couldn't just be the raw score.** Tried mapping `readmit_risk_score` directly to Legend for a continuous color gradient first. Power BI's core Scatter chart visual doesn't support this — the Legend field well only accepts categorical fields, never a continuous gradient, whether the source is a raw column (even with its "Default Summarization" property set to Average) or an explicit DAX measure (Power BI's own error: *"A single value for column cannot be determined... requires a non-measure field"*). This is a hard constraint of the visual, not a setting to find. The workaround is the same bucketing pattern used throughout this project (`pd.cut`, SQL `CASE WHEN`) — bin the continuous score into discrete tiers, then use *that* categorical column for color, and use the continuous score for bubble **Size** instead (Size *does* accept a continuous aggregation).

**The bucketing itself needed a second fix.** First attempt used fixed absolute cutoffs (`>= 0.6` High, `>= 0.3` Medium). Result: nearly every encounter came back "High," which looked like a bug but wasn't — it's the same calibration caveat already logged in Phase 4: `class_weight='balanced'` shifts the whole score distribution upward to compensate for the imbalanced target, so raw scores aren't literal probabilities and an absolute cutoff like "0.6" doesn't mean what it would for a normally-calibrated model. Fixed by switching to **percentile-based tiers** instead — top 20% of the *observed* score distribution = High, next 30% = Medium, bottom 50% = Low, computed via `PERCENTILEX.INC(ALL(risk_scores[readmit_risk_score]), risk_scores[readmit_risk_score], 0.8 / 0.5)` inside the bucketing formula. Percentile thresholds work regardless of where the raw scores sit, because they only rely on the scores' *relative ranking* — exactly the property the calibration caveat already said was trustworthy.

**A DAX mechanics gotcha hit along the way.** First attempt at the bucketing formula was written as a *measure*, which failed with "a single value for column cannot be determined" — measures only ever evaluate in aggregated/filter context, so DAX had no way to know which row's score to bucket. Rewriting the identical formula as a **calculated column** fixed it immediately, since calculated columns evaluate row-by-row (row context) at model refresh time. Conceptually the same distinction as `WHERE` vs `HAVING`, or `CASE WHEN` evaluating per-row in SQL — aggregated-context tools and row-context tools aren't interchangeable just because the syntax looks similar.

**What the tiered chart showed — a genuine finding, not a display bug.** After the percentile fix, tier averages separated cleanly (Low ≈ 0.44-0.53, Medium ≈ 0.57-0.62, High ≈ 0.68-0.76 average `readmit_risk_score`, confirmed via a pivot check). The chart still looks mostly "High" (blue) across most of the x-axis — but Low/Medium tiers cluster almost exclusively at `number_inpatient = 0`. This is the flagship Phase 2/4 finding restated visually: once a patient has *any* prior inpatient history, the model classifies them High risk almost regardless of emergency-visit count. Decided to keep the chart as-is rather than rebalance the tier cutoffs for visual variety — the "wall of blue" is an honest reflection of the underlying pattern, not a reason to disguise it.

**Deferred, not forgotten:** rename the project's variable/column names (SQL, Python, and Power BI) to be more self-descriptive/easier to track across the whole pipeline — flagged for a later cleanup pass, not part of Phase 5's functional build.

