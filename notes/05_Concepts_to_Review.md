# Concepts to Review (running list — mistakes made + new things learned)

*(Not analysis findings — those live in `02_EDA_Notes.md`. This is a skills/revision log.)*

## Pandas

## Seaborn / plotting

## Analytical reasoning — the recurring themes

**Why Python, not SQL, for parsing `IDS_mapping.csv`.** `SELECT`/`WHERE` only work on data that's already loaded into a well-structured table — they filter rows *within* an existing table, they don't fix a file that isn't a valid single table yet. `IDS_mapping.csv` isn't one table, it's three lookup tables (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) stacked in one file, each with its own header, separated by blank/comma-only lines. A generic import tool reads only the *first* header and treats everything after it as data for those same columns — so the second table's header row (`discharge_disposition_id,description`) would get imported as a literal garbage data row, not recognized as a new header. The table would never load cleanly enough to `SELECT` from in the first place. T-SQL could theoretically untangle this with string functions on a raw single-column import, but that's fighting the tool — it's the same line-by-line parsing logic Python's box/basket code already does, just far clunkier in SQL. General principle: **SQL is for querying data that's already well-structured. Python is for fixing structure that isn't clean yet.** This file needed the second thing before it could ever get to the first.

## SQL / database design

**Views vs. tables.** A table stores actual rows of data — frozen until something manually reloads it (e.g. `fact_encounter`, 99,343 rows, fixed until re-imported). A view stores a saved *query definition*, not data — it re-runs its underlying logic fresh every time it's queried, always reflecting whatever currently sits in the source tables. This is why `v_readmit_summary_by_diagnosis`'s output looked identical to running the CTE manually — it *is* the same computation, just saved under a name so it doesn't need retyping.

The real value of a view isn't that the output looks different — it's three things: reusability (`SELECT * FROM view_name` instead of retyping a full CTE/JOIN/GROUP BY every time); staying current automatically as source data changes (a saved copy of today's results would go stale the moment new encounters were added — a view never does); and giving tools like Power BI's DirectQuery a stable, named object to connect to, so every dashboard refresh re-runs the live query instead of showing a frozen snapshot.

Analogy: a view is like an Excel formula cell — recalculates every time it's opened. A table (or a `SELECT INTO` snapshot) is like a pasted static value — stays fixed until someone manually changes it.

**`CASE WHEN` — bucketing values (SQL's version of `pd.cut()`).** SQL has no direct equivalent to pandas' `pd.cut()`. `CASE WHEN ... THEN ... ELSE ... END` is the manual substitute — evaluated like a chain of if/elif checks per row, top to bottom, first matching condition wins. `ELSE` catches everything not explicitly matched (used as the open-ended `'3+'` bucket, mirroring `pd.cut`'s open-ended top bin).

**`WHERE` vs. `HAVING`.** `WHERE` filters individual rows *before* grouping happens, so it can't reference an aggregate like `COUNT(*)` or `AVG(...)` — those don't exist yet at that stage. `HAVING` filters *after* aggregation, specifically so it can reference aggregate values. A CTE is the alternative when the query is more complex than a single `HAVING` can express cleanly — aggregate first inside the CTE, then filter the CTE's own output with a normal `WHERE`.

**CTEs (`WITH ... AS (...)`).** A temporary, named result set usable like a table immediately after it's defined — lets you aggregate in one step and filter/query the aggregated result in the next, without nesting subqueries. Multiple CTEs can be chained in one `WITH` clause (comma-separated), each one able to reference any CTE defined before it (used in the utilization view — one CTE banded the raw counts, the next aggregated the banded result).

**Window functions (`OVER(...)`) — `RANK()` and `LAG()`.** Different from `GROUP BY`: a window function computes a value across a set of rows *without* collapsing them into fewer rows — it adds a new column while every row stays visible. `RANK() OVER (ORDER BY ...)` assigns a rank number; ties get the same rank and the next number is skipped (1,2,2,4...) — `DENSE_RANK()` doesn't skip (1,2,2,3...), `ROW_NUMBER()` never ties at all. `LAG(column) OVER (ORDER BY ...)` pulls in the *previous* row's value (per that ordering) into the current row — useful for row-to-row comparisons (e.g. gap to the next-highest category) without a self-join.

**`SUM()` can't take `BIT` directly.** Unlike Python, where `True`/`False` sum natively, SQL Server rejects `BIT` as a `SUM()` operand outright — needs an explicit `CAST(column AS INT)` first.

**Integer division truncates silently.** SQL Server performs integer division by default when both operands are integer types — without an explicit `CAST(... AS FLOAT)` somewhere in the calculation, a rate calculation would silently round every result to 0 or 1. No equivalent surprise in Python 3, where `/` always returns a float.

**Staging-table load pattern.** When an import tool can't load directly into an already-defined table (either because it can only create new tables, or because it mistypes columns from a small data sample), load into a throwaway staging table first, move the data with `INSERT INTO real_table SELECT * FROM staging_table`, then `DROP` the staging table. A standard, real-world ETL technique, not a workaround specific to this project.

**A batch doesn't stop after a runtime error, by default.** A compile-time error (bad syntax, unknown object) stops a script immediately. A runtime error (like a constraint violation) does not — by default, SQL Server keeps executing the remaining statements in the same batch. This is why a `DROP TABLE` written right after a failed `INSERT` still ran, deleting a staging table before it could be inspected. `SET XACT_ABORT ON`, or wrapping risky statements in an explicit transaction, changes this behavior.

## Power BI / DAX

**Scatter chart Legend only accepts categorical fields.** Unlike Values or Size, the Legend well on Power BI's core Scatter chart visual cannot render a continuous color gradient — not from a raw column (regardless of its "Default Summarization" setting) and not from a measure (Power BI blocks this explicitly: *"requires a non-measure field"*). To color-encode a continuous variable, bucket it into categories first (same principle as `pd.cut`/`CASE WHEN` elsewhere in this project) and put the bucketed category in Legend; put the raw continuous value in Size instead.

**Measures vs. calculated columns — aggregated context vs. row context.** A DAX *measure* only ever evaluates inside an aggregated/filter context (it computes a single number after filtering/grouping) — it cannot reference "the current row's value" the way a row-by-row bucketing formula needs to, and errors with "a single value... cannot be determined" if you try. A *calculated column* evaluates once per row at model refresh time (row context), which is what per-row bucketing (e.g. `SWITCH(TRUE(), ...)` tiers) actually needs. Same underlying distinction as `WHERE` (row-level) vs `HAVING` (aggregate-level) in SQL, just enforced more strictly by DAX with an explicit error instead of a silent wrong answer.

**Column "Default Summarization" property.** A model-level setting (Data view → select column → ribbon → Summarization dropdown) that controls how a column aggregates *by default* wherever it's newly placed in a visual — separate from Data type, and separate from a per-visual "Don't summarize" override already applied inside a specific well.

**"Sort by column" can't depend on the column it's sorting, even indirectly.** A calculated column built from `SWITCH()` on some text column (e.g., a manual order-number column derived from a category label) cannot then be set as that same category column's "Sort by column" target — Power BI raises an explicit circular dependency error, even though the actual values would resolve fine outside its own dependency tracker. Fix: define the order column at the data source (SQL) level instead of via DAX, so it's a genuinely independent column rather than one derived from the column it's meant to sort. (A plain `CASE WHEN`, or extracting a number directly out of an existing label with `SUBSTRING`/`CHARINDEX`, both work.)

**Absolute thresholds vs. percentile thresholds on an imbalance-corrected model's scores.** A model trained with `class_weight='balanced'` shifts its whole predicted-probability distribution to compensate for the imbalanced target — so a fixed cutoff like "score ≥ 0.6 = high risk" can misfire badly (e.g. classifying almost everyone as high risk) even though the scores are still valid for *relative ranking* (per the calibration caveat already logged in Phase 4). Percentile-based thresholds (top 20% = High, etc., via `PERCENTILEX.INC`) sidestep this because they only depend on relative ranking, not the raw score's absolute value — the same property that makes the scores trustworthy for a watchlist in the first place.

## Modeling (new for this project)

