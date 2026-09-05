# Hospital Readmissions & Quality Operations Project — Overview

**Roadmap slot:** Post-Week 7, Data Analyst 3-Month Roadmap (Project 2 of 2 new portfolio projects)
**Domain:** Healthcare / Hospital Operations
**Dataset:** [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ID 296; also mirrored on Kaggle)

---

## Why this project

- Diversifies the portfolio into a different healthcare data archetype than OASIS-2. OASIS-2 is a clinical/longitudinal cohort story (one patient tracked over time). This is hospital operations/quality data — encounter-level, cross-sectional, the kind an actual hospital analytics or quality-improvement team works with day to day.
- **Scope boundary (read this before Phase 1):** this project is framed as a hospital operations/quality-improvement question — readmission risk, length of stay, utilization patterns *within this hospital system*. It is deliberately **not** a payer/insurance-claims cost project (that's a separate, later project, using a different dataset). If a query or finding starts drifting toward "cost driver" or "claims" language, that's scope creep — pull it back to the operations framing.
- Directly reinforces the SQL patterns being drilled in interview prep (readmission-rate CTEs, GROUP BY/HAVING cohort queries, window functions) — this project is where that practice becomes a real, interpretable analysis instead of an isolated exercise.
- Adds a genuinely new skill this time instead of repeating OASIS-2: a lightweight predictive layer (logistic regression) on top of the descriptive/BI work, and optionally an Azure Data Factory ingestion pipeline instead of a manual SSMS import.

## Dataset at a glance

- 101,766 hospital encounters, 130 US hospitals and integrated delivery networks, 1999–2008
- Each row = one inpatient encounter for a patient with a diabetes diagnosis, stay length 1–14 days
- Features: demographics, admission/discharge/admission-source type, diagnoses (ICD-9), diabetic medications, number of lab procedures/medications, number of prior outpatient/emergency/inpatient visits, and the outcome column `readmitted` (`<30`, `>30`, `NO`)
- **Known from the source docs, confirm once loaded:** `patient_nbr` repeats — some patients have multiple encounters in the dataset. Decide encounter-level vs. patient-level unit of analysis in Phase 1, don't assume. (See `01_Dataset_Schema.md` — the paper this dataset comes from hit this exact issue and documented how they resolved it.)
- **Class balance to check early:** 30-day readmission is a minority outcome. The paper's own reduced subset (first encounter per patient only, hospice/death excluded) showed roughly 9-9.5%; the full 101,766-row file commonly distributed on Kaggle may run somewhat higher since it includes repeat encounters — compute your own number in Phase 1, don't assume either figure. Either way this matters for Phase 4 — accuracy alone will be a misleading metric.
- **Source paper now in the vault:** `reference/Strack_et_al_2014_HbA1c_Readmission.pdf` — Strack et al. 2014, the peer-reviewed paper this dataset was built for. Table 1 (feature list + real missingness rates) and Table 2 (the ICD-9 diagnosis grouping scheme) are now reflected in `01_Dataset_Schema.md`. Worth skimming the full paper before Phase 2 — their own findings (HbA1c testing was rare and associated with lower readmission) are a legitimate candidate angle for your own findings, or a useful comparison point if your data tells a different story.

## Business question

> Which encounter-level and utilization factors are associated with 30-day readmission, and what should a hospital quality-operations team flag or change in its discharge/follow-up process as a result?

## Deliverables

- [ ] Python analysis (pandas, seaborn) — cleaning, EDA, readmission-rate patterns
- [ ] Load cleaned data into Azure SQL DB (lean 2-table schema — revised Aug 1, her instinct: a repeat multi-table star schema after OASIS-2 shows nothing new) + CTE/window-function analytical queries + 3 reusable SQL views for Power BI to consume directly
- [ ] Python predictive layer — logistic regression on 30-day readmission, evaluated on precision/recall (not accuracy alone), top risk factors identified
- [ ] Power BI dashboard connected via SQL, DAX-driven measures, including a risk-flag view fed by the model output
- [ ] Stakeholder insight summary: 3 findings + 3 actions (hospital operations audience)
- [ ] Push to GitHub with README (matches OASIS-2 project pattern)
- [ ] Add to portfolio site alongside the other two projects

See `04_Project_Plan.md` for the phase-by-phase schedule.

## Status log

- **Jul 28:** Project scoped and vault created, mirroring the OASIS-2 structure at her request. Explicitly narrowed to hospital-operations framing only, after the first draft plan blended in payer/cost-analytics language that belongs to a separate future project.
- **Jul 31:** Source research paper (Strack et al. 2014) added to the vault. Dataset schema notes upgraded from UCI-doc placeholders to source-verified values, including the exact ICD-9 diagnosis grouping scheme and the unit-of-analysis precedent — real groundwork for Phase 1, not yet started.
