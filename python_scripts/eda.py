from pathlib import Path
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

@dataclass
class eda:
    project_root: Path
    plot_dir: Path
    df: pd.DataFrame | None = field(default=None, init=False)

    def save_plot(self, fig: plt.Figure, file_name: str, dpi: int = 150):
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.plot_dir / file_name
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def readmit_summary(self, data: pd.DataFrame, grp_col: str | list[str], out_col: str = "patient_readmit",z_score: float = 1.96,) -> pd.DataFrame:
        summary_table = (
                data.groupby(grp_col)[out_col].agg(p="mean", n="count").assign
                (rate=lambda d: d["p"] * 100,
                margin_of_error=lambda d: z_score * np.sqrt(d["p"] * (1 - d["p"]) / d["n"]) * 100)
                .drop(columns="p").round({"rate": 2, "margin_of_error": 2})
            )
        return summary_table
    
    @staticmethod
    def categorize_diagnosis(code: object) -> str:
        if pd.isna(code):
            return "Missing"

        diagnosis_code_text = str(code).strip()
        if diagnosis_code_text.startswith(("V", "E")):
            return "Other"
        try:
            diagnosis_code_number = int(float(diagnosis_code_text))
        except ValueError:
            return "Other"

        diagnosis_code_to_category_map = {
            250: "Diabetes",
            785: "Circulatory",
            786: "Respiratory",
            787: "Digestive",
            788: "Genitourinary",
        }

        if diagnosis_code_number in diagnosis_code_to_category_map:
            return diagnosis_code_to_category_map[diagnosis_code_number]

        code_range_to_category = [
            ((390, 459), "Circulatory"),
            ((460, 519), "Respiratory"),
            ((520, 579), "Digestive"),
            ((580, 629), "Genitourinary"),
            ((710, 739), "Musculoskeletal"),
            ((800, 999), "Injury"),
            ((140, 239), "Neoplasms"),
        ]

        for (range_start, range_end), category_name in code_range_to_category:
            if range_start <= diagnosis_code_number <= range_end:
                return category_name
        return "Other"

    def load_and_clean_data(self) -> pd.DataFrame:
        data_path = self.project_root / "diabetic_data.csv"
        drop_columns = ["weight", "payer_code"]
        exclude_discharge_ids = {11, 13, 14, 19, 20, 21}

        data = pd.read_csv(data_path,na_values=["?"], keep_default_na=True)
        data = (data.drop(columns=drop_columns, errors="ignore").assign
                (medical_specialty=lambda table: table["medical_specialty"].fillna("Missing"), patient_readmit=lambda table: table["readmitted"].eq("<30")).loc
                [lambda table: ~table["discharge_disposition_id"].isin(exclude_discharge_ids)].copy())
        self.df = data
        return data

    def line_plot(self, data: pd.DataFrame, x: str, y: str, hue: str, xlabel: str, ylabel: str, title: str, file_name: str, rotate_x: bool = False, legend_title: str | None = None,):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x=x, y=y, hue=hue, marker="o", palette="tab10", ax=ax,)
        if rotate_x:
            ax.tick_params(axis="x", rotation=45)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
        self.save_plot(fig, file_name)

    def generate_eda_outputs(self):
        data = self.require_df()
        
        self.add_eda_bands(data)
        self.plot_readmission_by_age_and_meds(data)
        self.plot_readmission_by_hospital_stay_and_meds(data)
        self.plot_readmission_by_previous_hospital_and_er_visits(data)

        self.add_age_numeric(data)
        self.plot_correlation_heatmap(data)
        self.plot_effect_sizes()
        self.plot_readmission_odds_by_risk_factor()

        self.add_diagnosis_category(data)
        self.plot_diabetes_readmission_by_age_group(data)

    def require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data is not loaded. Call load_and_clean_data first.")
        return self.df

    def add_eda_bands(self, data: pd.DataFrame):
        data["med_count_band"] = pd.cut(data["num_medications"],
            bins=[0, 5, 10, 15, 20, 25, 30, 100],
            labels=["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31+"])
        data["stay_band"] = pd.cut(data["time_in_hospital"],
            bins=[0, 3, 6, 9, 14],
            labels=["1-3", "4-6", "7-9", "10-14"])
        data["prior_inpatient_band"] = pd.cut(data["number_inpatient"],
            bins=[-1, 0, 1, 2, 100],
            labels=["0", "1", "2", "3+"])
        data["prior_emergency_band"] = pd.cut(data["number_emergency"],
            bins=[-1, 0, 1, 2, 100],
            labels=["0", "1", "2", "3+"])

    def plot_readmission_by_age_and_meds(self, data: pd.DataFrame):
        summary_table = self.readmit_summary(data, ["age", "med_count_band"]).reset_index()
        reliable_rows = summary_table[summary_table["n"] >= 30]
        self.line_plot(
            data=reliable_rows,
            x="age",
            y="rate",
            hue="med_count_band",
            xlabel="Age band",
            ylabel="Readmission rate (%)",
            title="Readmission rate by age, split by number of medications at discharge\n(only combinations with n >= 30 shown)",
            file_name="readmission_by_age_and_medication.png",
            legend_title="Medications",
        )

    def plot_readmission_by_hospital_stay_and_meds(self, data: pd.DataFrame):
        summary_table = self.readmit_summary(data, ["stay_band", "med_count_band"]).reset_index()
        reliable_rows = summary_table[summary_table["n"] >= 100]
        self.line_plot(
            data=reliable_rows,
            x="stay_band",
            y="rate",
            hue="med_count_band",
            xlabel="Length of stay (days)",
            ylabel="Readmission rate (%)",
            title="Readmission rate by stay length, split by medications at discharge\n(only combinations with n >= 100 shown)",
            file_name="readmission_by_hospital_stay_and_medication.png",
            legend_title="Medications",
        )

    def plot_readmission_by_previous_hospital_and_er_visits(self, data: pd.DataFrame):
        summary_table = self.readmit_summary(data, ["prior_inpatient_band", "prior_emergency_band"]).reset_index()
        reliable_rows = summary_table[summary_table["n"] >= 30].sort_values("rate", ascending=False)
        self.line_plot(
            data=reliable_rows,
            x="prior_inpatient_band",
            y="rate",
            hue="prior_emergency_band",
            xlabel="Prior inpatient visits (banded)",
            ylabel="Readmission rate (%)",
            title="Readmission rate by prior inpatient visits, split by prior emergency visits\n(only combinations with n >= 30 shown)",
            file_name="readmission_by_previous_hospital_and_er_visits.png",
            legend_title="Prior ER visits",
        )

    def add_age_numeric(self, data: pd.DataFrame):
        data["age_numeric"] = (data["age"].str.strip("[)").str.split("-").apply(lambda x: (int(x[0]) + int(x[1])) / 2))

    def plot_correlation_heatmap(self, data: pd.DataFrame):
        correlation_columns = [
            "age_numeric",
            "time_in_hospital",
            "num_medications",
            "number_inpatient",
            "number_emergency",
            "patient_readmit",
        ]
        correlation_matrix = data[correlation_columns].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Correlation between key readmission risk factors")
        self.save_plot(fig, "correlation_heatmap.png")

    def plot_effect_sizes(self) -> None:
        effect_sizes = pd.DataFrame(
            {"variable": ["Prior inpatient visits", "Prior emergency visits", "Length of stay", "Number of medications", "Age"],
            "swing_pp": [17.8, 14.6, 6.4, 5.4, 2.8]}).sort_values("swing_pp")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(effect_sizes["variable"], effect_sizes["swing_pp"])
        ax.set_xlabel("Readmission rate swing (percentage points)")
        ax.set_title("Which factor moved readmission rate the most?")
        self.save_plot(fig, "eda_effect_size.png")

    def plot_readmission_odds_by_risk_factor(self):
        odds_ratios = pd.DataFrame(
            {"finding": ["Combined (flagship)", "Prior inpatient", "Prior emergency", "Length of stay", "Number of medications", "Age"],
            "odds_ratio": [5.70, 3.82, 2.83, 1.90, 1.82, 1.33]}).sort_values("odds_ratio")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(odds_ratios["finding"], odds_ratios["odds_ratio"])
        ax.axvline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Odds ratio vs. reference group")
        ax.set_title("Readmission odds by risk factor")
        self.save_plot(fig, "eda_odds_ratio.png")

    def add_diagnosis_category(self, data: pd.DataFrame):
        data["diagnosis_category"] = data["diag_1"].apply(self.categorize_diagnosis)

    def plot_diabetes_readmission_by_age_group(self, data: pd.DataFrame):
        summary_table = self.readmit_summary(data, ["age", "diagnosis_category"]).reset_index()
        reliable_rows = summary_table[summary_table["n"] >= 30]
        diabetes_by_age = reliable_rows[reliable_rows["diagnosis_category"] == "Diabetes"].sort_values("age")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(diabetes_by_age["age"], diabetes_by_age["rate"])
        ax.axhline(11.4, color="gray", linestyle="--", linewidth=1, label="Overall baseline (11.4%)")
        ax.set_xlabel("Age band")
        ax.set_ylabel("Readmission rate (%) - diabetes-primary encounters")
        ax.set_title("Readmission rate for diabetes as primary diagnoses by age")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        self.save_plot(fig, "eda_diabetes_by_age.png")

    def build_and_export_dim_fact(self):
        data = self.require_df()
        data = self.df

        diagnosis_table = (data[["diag_1", "diagnosis_category"]].drop_duplicates().reset_index(drop=True))
        diagnosis_table["diagnosis_key"] = diagnosis_table.index + 1
        diagnosis_table = diagnosis_table.rename(columns={"diag_1": "diag_1_raw"})[["diagnosis_key", "diag_1_raw", "diagnosis_category"]]

        data_with_diagnosis_key = data.merge(diagnosis_table[["diag_1_raw", "diagnosis_key"]],left_on="diag_1", right_on="diag_1_raw", how="left").drop(columns=["diag_1_raw"])

        columns_to_keep = [
            "encounter_id",
            "patient_nbr",
            "race",
            "gender",
            "age",
            "medical_specialty",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "time_in_hospital",
            "num_medications",
            "num_lab_procedures",
            "num_procedures",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "diagnosis_key",
            "readmitted",
        ]
        encounter_table = data_with_diagnosis_key[columns_to_keep].copy()
        encounter_table["readmit_30"] = (encounter_table["readmitted"] == "<30").astype(int)

        encounter_table.to_csv(self.project_root / "fact_encounter.csv", index=False)
        diagnosis_table.to_csv(self.project_root / "dim_diagnosis.csv", index=False)
        self.df = data_with_diagnosis_key

    def run(self):
        self.load_and_clean_data()
        self.generate_eda_outputs()
        self.build_and_export_dim_fact()
