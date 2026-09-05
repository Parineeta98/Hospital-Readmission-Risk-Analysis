-- dim_diagnosis (lookup) + fact_encounter (main table), linked by diagnosis_key.
-- Lookup table: one row per unique raw ICD-9 code, tagged with its
-- clinical category (via categorize_diagnosis() in Python, using the
-- Strack et al. 2014 grouping scheme).

create table dim_diagnosis (
    diagnosis_key INT PRIMARY KEY,
    diag_1_raw VARCHAR(10),
    diagnosis_category VARCHAR(30)
);

-- Main encounter-level table. diagnosis_key is a foreign key back to dim_diagnosis

create table fact_encounter (
    encounter_id INT PRIMARY KEY,
    patient_nbr INT,
    race VARCHAR(30),
    gender VARCHAR(20),
    age VARCHAR(10),
    medical_specialty VARCHAR(50),
    admission_type_id INT,
    discharge_disposition_id INT,
    admission_source_id INT,
    time_in_hospital INT,
    num_medications INT,
    num_lab_procedures INT,
    num_procedures INT,
    number_outpatient INT,
    number_emergency INT,
    number_inpatient INT,
    number_diagnoses INT,
    diagnosis_key INT,
    readmitted VARCHAR(5),
    readmit_30 BIT,
    FOREIGN KEY (diagnosis_key) REFERENCES dim_diagnosis(diagnosis_key)
);

create table risk_scores (
    encounter_id INT PRIMARY KEY,
    readmit_risk_score FLOAT,
    FOREIGN KEY (encounter_id) REFERENCES fact_encounter(encounter_id)
);

-- needed to create dupe files first since import data wizard was not available
-- dupe tables dropped and data inserted into orignal tables

insert into dim_diagnosis
select * from dim_diagnosis_dupe;

drop table dim_diagnosis_dupe;

select count(*) from dim_diagnosis;

-- Needed a sanity check before loading fact_encounter to confirm every diagnosis_key 
-- in the staged data actually exists in dim_diagnosis.
-- Error expected to be raised here instead of hitting an FK violation mid-insert.
select distinct fe.diagnosis_key
from fact_encounter_dupe fe
left join dim_diagnosis dd on fe.diagnosis_key = dd.diagnosis_key
where dd.diagnosis_key is null and fe.diagnosis_key is not null;
-- if all correct, should return 0 rows back

insert into fact_encounter
select * from fact_encounter_dupe;

drop table fact_encounter_dupe;

--final check of row counts (should get fact_encounter = 99,343 and dim_diagnosis = 716)
select count(*) from fact_encounter;
select count(*) from dim_diagnosis;

-- Analysis 1: Readmission rate by diagnosis category x age band.
-- Recreating Python finding that resolved the [20-30) age spike: diabetes-primary encounters 
-- specifically show an elevated ~20% readmission rate in that age band, well above every other age group. 
-- This confirms the result in both SQL and Python.

with readmit_by_diag_age as (
    select
        dd.diagnosis_category,
        fe.age,
        count(*) as n,                                                        
        sum(cast(fe.readmit_30 as INT)) as readmit_count,                     
        cast(sum(cast(fe.readmit_30 as INT)) as FLOAT) / count(*) * 100 as readmit_rate  
    from fact_encounter fe
    join dim_diagnosis dd on fe.diagnosis_key = dd.diagnosis_key              
    group by dd.diagnosis_category, fe.age
)
select *
from readmit_by_diag_age
where n >= 30 -- same as Python
order by diagnosis_category, age;

-- this analysis in redone in SQL because one, to cross-check and two, to put the same logic in Power BI and querying it live. 

-- Analysis 2: Average length of stay by diagnosis category.

select
    dd.diagnosis_category,
    count(*) as n,
    avg(cast(fe.time_in_hospital as FLOAT)) as avg_length_of_stay_in_days
from fact_encounter fe
join dim_diagnosis dd on fe.diagnosis_key = dd.diagnosis_key
group by dd.diagnosis_category
having count(*) >= 30  
order by avg_length_of_stay_in_days desc;

-- Result: Neoplasms has the longest average stay (5.28 days), its clinically 
-- sensible as cancer care involves more complex treatment than the other categories.

-- Analysis 3: Readmission rate by prior inpatient x prior emergency visits

with banded as (
    select
        case
            when number_inpatient = 0 then '0'
            when number_inpatient = 1 then '1'
            when number_inpatient = 2 then '2'
            else '3+'
        end as prior_inpatient_band,
        case
            when number_emergency = 0 then '0'
            when number_emergency = 1 then '1'
            when number_emergency = 2 then '2'
            else '3+'
        end as prior_emergency_band,
        readmit_30
    from fact_encounter
)
select
    prior_inpatient_band,
    prior_emergency_band,
    count(*) as n,
    cast(sum(cast(readmit_30 as INT)) as FLOAT) / count(*) * 100 as readmit_rate
from banded
group by prior_inpatient_band, prior_emergency_band
having count(*) >= 30
order by readmit_rate desc;

-- Result: match with Python: more than 3 times inpatient and 2 emergency visit has the
-- highest readmission rate (34.6%) vs 8.45% baseline at 0x0 (n=61,741)

-- Analysis 4: Ranking diagnosis categories by readmission rate.

with diag_rates as (
    select
        dd.diagnosis_category,
        count(*) as n,
        cast(sum(cast(fe.readmit_30 as INT)) as FLOAT) / count(*) * 100 as readmit_rate
    from fact_encounter fe
    join dim_diagnosis dd on fe.diagnosis_key = dd.diagnosis_key
    group by dd.diagnosis_category
    having count(*) >= 30
)
select
    diagnosis_category,
    n,
    readmit_rate,
    rank() over (order by readmit_rate desc, n desc) as rate_rank
from diag_rates
order by rate_rank;

-- Result: excluded Missing (n=20, so unreliable), Diabetes ranks highest at 13.10% (n=8661), 
-- Musculoskeletal lowest at 9.54%. Diabetes topping the list is consistent with the earlier finding that 
-- diabetes-primary diagnoses drive elevated risk, especially in the [20-30) age band.

-- Analysis 5: Gap between each diagnosis category and the one ranked directly above it.

with diag_rates as (
    select
        dd.diagnosis_category,
        count(*) as n,
        cast(sum(cast(fe.readmit_30 as INT)) as FLOAT) / count(*) * 100 as readmit_rate
    from fact_encounter fe
    join dim_diagnosis dd on fe.diagnosis_key = dd.diagnosis_key
    group by dd.diagnosis_category
    having count(*) >= 30
),

diag_rates_with_lag as (
    select
        diagnosis_category,
        n,
        readmit_rate,
        lag(readmit_rate) over (order by readmit_rate desc) as next_higher_rate
    from diag_rates
)
select
    diagnosis_category,
    n,
    readmit_rate,
    next_higher_rate,
    readmit_rate - next_higher_rate as gap_from_next_higher,
    readmit_rate - 11.4 as gap_from_overall_baseline

from diag_rates_with_lag
order by readmit_rate desc;

-- Result: only 4 of 9 categories sit above the dataset's 11.4% baseline (Diabetes
-- +1.70percent pt) the other 5 sit below it, down to Musculoskeletal at -1.86percent pt.

-- creating better views of the result tables
-- View 1: readmit_summary() by diagnosis category.
-- View 2: readmit_summary(), by age band.
-- Same pattern as view 1 -- rate, n, margin_of_error -- but grouped by
-- age instead of diagnosis. No JOIN needed, age is already a column
-- on fact_encounter.

-- View 1: readmit_summary() by age band.

create view v_readmit_summary_by_age as
with base as (
    select
        age,
        count(*) as n,
        cast(sum(cast(readmit_30 as INT)) as FLOAT) / count(*) as rate_decimal
    from fact_encounter
    group by age
)
select
    age,
    n,
    rate_decimal * 100 as readmit_rate,
    1.96 * SQRT(rate_decimal * (1 - rate_decimal) / n) * 100 as margin_of_error
from base;

create view v_readmit_summary_by_diagnosis as
with base as (
    select
        dd.diagnosis_category,
        count(*) as n,
        cast(sum(cast(fe.readmit_30 as INT)) as FLOAT) / count(*) as rate_decimal
    from fact_encounter fe
    join dim_diagnosis dd on fe.diagnosis_key = dd.diagnosis_key
    group by dd.diagnosis_category
)
select
    diagnosis_category,
    n,
    rate_decimal * 100 as readmit_rate,
    1.96 * SQRT(rate_decimal * (1 - rate_decimal) / n) * 100 as margin_of_error
from base;
-- just checking
select * from v_readmit_summary_by_diagnosis order by readmit_rate desc;
select * from v_readmit_summary_by_age order by age;

-- View 2: readmit_summary() by prior inpatient x prior emergency band.

create view v_readmit_summary_by_utilization as
with banded as (
    select
        case
            when number_inpatient = 0 then '0'
            when number_inpatient = 1 then '1'
            when number_inpatient = 2 then '2'
            else '3+'
        end as prior_inpatient_band,
        case
            when number_emergency = 0 then '0'
            when number_emergency = 1 then '1'
            when number_emergency = 2 then '2'
            else '3+'
        end as prior_emergency_band,
        readmit_30
    from fact_encounter
),
base as (
    select
        prior_inpatient_band,
        prior_emergency_band,
        count(*) as n,
        cast(sum(cast(readmit_30 as INT)) as FLOAT) / count(*) as rate_decimal
    from banded
    group by prior_inpatient_band, prior_emergency_band
)
select
    prior_inpatient_band,
    prior_emergency_band,
    n,
    rate_decimal * 100 as readmit_rate,
    1.96 * SQRT(rate_decimal * (1 - rate_decimal) / n) * 100 as margin_of_error
from base;
--just checking
select * from v_readmit_summary_by_utilization order by readmit_rate desc;

-- risk encounters scores -- needed to create a dupe file again and was then dropped

insert into risk_scores
select * from risk_scores_dupe;

drop table risk_scores_dupe;

select count(*) from risk_scores;

-- sanity check 
select count(*)
from fact_encounter fe
left join risk_scores rs on fe.encounter_id = rs.encounter_id
where rs.encounter_id is null;

ALTER TABLE fact_encounter ADD is_first_encounter BIT;

-- ROW_NUMBER() partitioned by patient, ordered by encounter_id, ranks each patient's encounters in order 
with ranked ascan yo (
    select encounter_id,
           row_number() over (partition by patient_nbr order by encounter_id) as rn
    from fact_encounter
)
update fe
set fe.is_first_encounter = case when r.rn = 1 then 1 else 0 end
from fact_encounter fe
join ranked r on fe.encounter_id = r.encounter_id;

select sum(cast(is_first_encounter as int)) from fact_encounter;

select
    admission_source_id,
    count(*) as n,
    cast(sum(cast(readmit_30 as int)) as float) / count(*) * 100 as readmit_rate
from fact_encounter
group by admission_source_id
having count(*) >= 30
order by readmit_rate desc;

create table dim_admission_source (
    admission_source_id INT PRIMARY KEY,
    description VARCHAR(100)
);

insert into dim_admission_source
select * from dim_admission_source_dupe;

drop table dim_admission_source_dupe;

select count(*) from dim_admission_source;

alter table fact_encounter
add constraint FK_admission_source FOREIGN KEY (admission_source_id) REFERENCES dim_admission_source(admission_source_id);

-- View 4: readmit_summary(), by admission source. Same pattern as the
-- other three views (rate, n, margin_of_error), joined to
-- dim_admission_source for readable labels. Junk placeholder codes
-- (Not Available x2, NULL, Not Mapped, Unknown/Invalid) collapsed into
-- one "Unknown / Not Recorded" bucket -- they're statistically fine
-- (decent n) but semantically meaningless as individual categories.

create view v_readmit_summary_by_admission_source as
with base as (
    select
        case
            when das.description in ('Not Available', 'NULL', 'Not Mapped', 'Unknown/Invalid') then 'Unknown / Not Recorded'
            else das.description
        end as admission_source,
        count(*) as n,
        cast(sum(cast(fe.readmit_30 as INT)) as FLOAT) / count(*) as rate_decimal
    from fact_encounter fe
    join dim_admission_source das on fe.admission_source_id = das.admission_source_id
    group by case
            when das.description in ('Not Available', 'NULL', 'Not Mapped', 'Unknown/Invalid') then 'Unknown / Not Recorded'
            else das.description
        end
)
select
    admission_source,
    n,
    rate_decimal * 100 as readmit_rate,
    1.96 * SQRT(rate_decimal * (1 - rate_decimal) / n) * 100 as margin_of_error
from base;

select * from v_readmit_summary_by_admission_source order by readmit_rate desc;

---cross checking analysis with the paper-recreating them
select count(*) from hba1c_diagnosis_predictions;

alter table hba1c_diagnosis_predictions add hba1c_order INT;
GO

update hba1c_diagnosis_predictions
set hba1c_order = case hba1c_category
    when 'Not measured' then 1
    when 'Normal' then 2
    when 'High, changed' then 3
    when 'High, not changed' then 4
end;

--for filtering first-enciunter only
create view v_readmit_summary_by_age_first_encounter as
with base as (
    select
        age,
        count(*) as n,
        cast(sum(cast(readmit_30 as INT)) as FLOAT) / count(*) as rate_decimal
    from fact_encounter
    where is_first_encounter = 1
    group by age
)
select
    age,
    n,
    rate_decimal * 100 as readmit_rate,
    1.96 * SQRT(rate_decimal * (1 - rate_decimal) / n) * 100 as margin_of_error
from base;
GO
--check
select * from v_readmit_summary_by_age_first_encounter order by age;