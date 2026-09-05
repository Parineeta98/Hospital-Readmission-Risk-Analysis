from pathlib import Path
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

@dataclass
class recreateplotsfromstrack:
    project_root: Path
    data_dir: Path
    plot_dir: Path

    data: pd.DataFrame | None = field(default=None, init=False)
    df_model: pd.DataFrame | None = field(default=None, init=False)
    analysis_data: pd.DataFrame | None = field(default=None, init=False)

    def save_plot(self, fig: plt.Figure, file_name: str, dpi: int = 150):
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.plot_dir / file_name
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def load_data(self):
        patient_encounter_data = pd.read_csv(self.data_dir / "diabetic_data.csv", na_values=["?"])
        patient_encounter_data["readmit_30"] = (patient_encounter_data["readmitted"] == "<30").astype(int)

        if "diagnosis_category" not in patient_encounter_data.columns:
            encounter_to_diagnosis_key = pd.read_csv(self.data_dir / "fact_encounter.csv", usecols=["encounter_id", "diagnosis_key"])
            diagnosis_key_to_category = pd.read_csv(self.data_dir / "dim_diagnosis.csv", usecols=["diagnosis_key", "diagnosis_category"])
            diagnosis_category_by_encounter = (encounter_to_diagnosis_key.merge( diagnosis_key_to_category, on="diagnosis_key", how="left", validate="many_to_one")
                [["encounter_id", "diagnosis_category"]].drop_duplicates(subset="encounter_id", keep="first"))
            patient_encounter_data = patient_encounter_data.merge(diagnosis_category_by_encounter, on="encounter_id", how="left")

        patient_encounter_data["diagnosis_category"] = (patient_encounter_data["diagnosis_category"].fillna("Missing"))
        self.data = patient_encounter_data
        
    @staticmethod
    def recode_hba1c(row: pd.Series) -> str:
        if row["A1Cresult"] == "None":
            return "Not measured"
        if row["A1Cresult"] == "Norm":
            return "Normal"
        return "High, changed" if row["change"] == "Ch" else "High, not changed"
    
    @staticmethod
    def recode_race(race_value: object) -> str:
        if pd.isna(race_value):
            return "Missing"
        if race_value == "Caucasian":
            return "Caucasian"
        if race_value == "AfricanAmerican":
            return "African American"
        return "Other"

    @staticmethod
    def recode_admission(admission_source: object) -> str:
        if admission_source == 7:
            return "Emergency"
        if admission_source in [1, 2, 3]:
            return "Referral"
        return "Other"

    @staticmethod
    def recode_specialty(specialty: object) -> str:
        if specialty == "Cardiology":
            return "Cardiology"
        if specialty == "Family/GeneralPractice":
            return "General practice"
        if specialty == "InternalMedicine":
            return "Internal medicine"
        if specialty == "Missing":
            return "Missing"
        specialty_str = str(specialty)
        if "Surg" in specialty_str:
            return "Surgery"
        return "Other"

    @staticmethod
    def recode_age3(age_band: str) -> str:
        if age_band in ["[0-10)", "[10-20)", "[20-30)"]:
            return "<30"
        if age_band in ["[60-70)", "[70-80)", "[80-90)", "[90-100)"]:
            return "[60,100)"
        return "[30,60)"

    def recreate_paper_plots_and_export(self) -> None:
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data first.")

        df_model = (self.data.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first").copy())
        age_rates = df_model.groupby("age")["readmit_30"].mean()
        logit_rates = np.log(age_rates / (1 - age_rates))
        figure = plt.figure(figsize=(9, 5))
        plt.plot(age_rates.index, logit_rates.values, marker="o", color="black")
        plt.xlabel("Age (years)")
        plt.ylabel("Logit of probability of readmission")
        plt.title("Figure 2 recreation: Age vs. logit of readmission rate")
        plt.xticks(rotation=45)
        self.save_plot(figure, "paper_figure2_age_logit.png")

        raw = pd.read_csv(self.data_dir / "diabetic_data.csv", keep_default_na=False, na_values=["?"])
        raw_hba1c = raw[["encounter_id", "A1Cresult", "change"]]
        df_model = df_model.drop(columns=["A1Cresult", "change"], errors="ignore").merge(raw_hba1c, on="encounter_id", how="left")
        df_model["hba1c_category"] = df_model.apply(self.recode_hba1c, axis=1)
        raw_covariates = raw[["encounter_id", "race", "discharge_disposition_id", "admission_source_id"]]
        df_model = df_model.drop(
            columns=["race", "discharge_disposition_id", "admission_source_id"],
            errors="ignore").merge(raw_covariates, on="encounter_id", how="left")
        df_model["race_category"] = df_model["race"].apply(self.recode_race)
        df_model["discharge_category"] = df_model["discharge_disposition_id"].apply(lambda d: "Home" if d == 1 else "Other")
        df_model["admission_category"] = df_model["admission_source_id"].apply(self.recode_admission)
        df_model["specialty_category"] = df_model["medical_specialty"].apply(self.recode_specialty)
        df_model["age_category"] = df_model["age"].apply(self.recode_age3)

        formula = (
            "readmit_30 ~ time_in_hospital "
            "+ C(discharge_category, Treatment(reference='Home')) "
            "+ C(race_category, Treatment(reference='African American')) "
            "+ C(admission_category, Treatment(reference='Emergency')) "
            "+ C(specialty_category, Treatment(reference='Cardiology')) "
            "+ C(age_category, Treatment(reference='[30,60)')) "
            "+ C(diagnosis_category, Treatment(reference='Diabetes')) "
            "* C(hba1c_category, Treatment(reference='Not measured'))")
        
        df_model_clean = df_model[df_model["diagnosis_category"] != "Missing"].copy()
        adjusted_model = smf.glm(formula=formula, data=df_model_clean, family=sm.families.Binomial()).fit()
        mean_time_in_hospital = df_model_clean["time_in_hospital"].mean()
        hba1c_categories = ["Not measured", "Normal", "High, changed", "High, not changed"]

        diagnosis_categories_fig1 = ["Diabetes", "Respiratory", "Circulatory"]
        pred_df = self.build_prediction_df(
            diagnosis_categories=diagnosis_categories_fig1,
            hba1c_categories=hba1c_categories,
            mean_time_in_hospital=mean_time_in_hospital)
        
        pred_df = self.add_predictions(adjusted_model, pred_df)
        self.plot_figure1(pred_df, hba1c_categories)

        diagnosis_categories_all = [
            "Diabetes",
            "Circulatory",
            "Digestive",
            "Genitourinary",
            "Injury",
            "Musculoskeletal",
            "Neoplasms",
            "Other",
            "Respiratory",
        ]
        pred_df_fig3 = self.build_prediction_df(
            diagnosis_categories=diagnosis_categories_all,
            hba1c_categories=hba1c_categories,
            mean_time_in_hospital=mean_time_in_hospital)
        
        pred_df_fig3 = self.add_predictions(adjusted_model, pred_df_fig3)
        self.plot_figure3(pred_df_fig3, hba1c_categories)

        cell_counts = (df_model_clean.groupby(["diagnosis_category", "hba1c_category"]).size().unstack(fill_value=0))
        core_three = ["Diabetes", "Circulatory", "Respiratory"]
        panel_a = ["Diabetes", "Circulatory", "Respiratory", "Digestive", "Other"]
        pred_df_fig3["is_figure1_category"] = pred_df_fig3["diagnosis_category"].isin(core_three)
        pred_df_fig3["panel"] = pred_df_fig3["diagnosis_category"].apply(lambda d: "A" if d in panel_a else "B")
        pred_df_fig3["n"] = pred_df_fig3.apply(lambda row: cell_counts.loc[row["diagnosis_category"], row["hba1c_category"]], axis=1)
        export_cols = [
            "diagnosis_category",
            "hba1c_category",
            "predicted_rate",
            "ci_lower",
            "ci_upper",
            "n",
            "is_figure1_category",
            "panel"]
        paper_recreation_export = pred_df_fig3[export_cols].copy()
        paper_recreation_export.to_csv(self.data_dir / "hba1c_diagnosis_predictions.csv", index=False)

    def build_prediction_df(self, diagnosis_categories: list[str], hba1c_categories: list[str], mean_time_in_hospital: float) -> pd.DataFrame:
        pred_rows: list[dict[str, object]] = []
        for diagnosis in diagnosis_categories:
            for hba1c in hba1c_categories:
                pred_rows.append(
                    {
                        "time_in_hospital": mean_time_in_hospital,
                        "discharge_category": "Home",
                        "race_category": "African American",
                        "admission_category": "Emergency",
                        "specialty_category": "Cardiology",
                        "age_category": "[30,60)",
                        "diagnosis_category": diagnosis,
                        "hba1c_category": hba1c,
                    })
        return pd.DataFrame(pred_rows)

    @staticmethod
    def add_predictions(model: sm.GLM, pred_df: pd.DataFrame) -> pd.DataFrame:
        pred_summary = model.get_prediction(pred_df).summary_frame(alpha=0.05)
        pred_df = pred_df.copy()
        pred_df["predicted_rate"] = pred_summary["mean"]
        pred_df["ci_lower"] = pred_summary["mean_ci_lower"]
        pred_df["ci_upper"] = pred_summary["mean_ci_upper"]
        return pred_df

    def plot_figure1(self, prediction_table: pd.DataFrame, hba1c_order: list[str]):
        diagnosis_colors = {"Diabetes": "#2a78d6", "Respiratory": "#2ca02c","Circulatory": "#d62728"}
        diagnosis_names = ["Diabetes", "Respiratory", "Circulatory"]
        figure = plt.figure(figsize=(9, 6))

        for diagnosis_name in diagnosis_names:
            diagnosis_subset = (prediction_table[prediction_table["diagnosis_category"] == diagnosis_name].set_index("hba1c_category").loc[hba1c_order])
            lower_error = (diagnosis_subset["predicted_rate"] - diagnosis_subset["ci_lower"])
            upper_error = (diagnosis_subset["ci_upper"] - diagnosis_subset["predicted_rate"])
            plt.errorbar(hba1c_order, diagnosis_subset["predicted_rate"], yerr=[lower_error, upper_error], marker="o", label=diagnosis_name, color=diagnosis_colors[diagnosis_name],capsize=4)

        plt.xlabel("HbA1c")
        plt.ylabel("Probability of readmission")
        plt.title("Figure 1 recreation: Adjusted readmission rate by diagnosis and HbA1c")
        plt.legend()
        self.save_plot(figure, "paper_figure1.png")

    def plot_figure3(self, prediction_table: pd.DataFrame, hba1c_order: list[str]):
        panel_a_diagnoses = ["Diabetes", "Circulatory", "Respiratory", "Digestive", "Other"]
        panel_b_diagnoses = ["Genitourinary", "Injury", "Musculoskeletal", "Neoplasms"]

        figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        for axis, diagnosis_names, panel_title in [
            (axes[0], panel_a_diagnoses, "Panel A - larger, well-powered categories"),
            (axes[1], panel_b_diagnoses, "Panel B - smaller categories, wider uncertainty"),]:
            for diagnosis_name in diagnosis_names:
                diagnosis_subset = (
                    prediction_table[prediction_table["diagnosis_category"] == diagnosis_name].set_index("hba1c_category").loc[hba1c_order])
                lower_error = (diagnosis_subset["predicted_rate"] - diagnosis_subset["ci_lower"])
                upper_error = (diagnosis_subset["ci_upper"] - diagnosis_subset["predicted_rate"])
                axis.errorbar(
                    hba1c_order,
                    diagnosis_subset["predicted_rate"],
                    yerr=[lower_error, upper_error],
                    marker="o",
                    label=diagnosis_name,
                    capsize=3)
            axis.set_xlabel("HbA1c")
            axis.set_title(panel_title)
            axis.legend(fontsize=8)
            axis.tick_params(axis="x", rotation=20)

        axes[0].set_ylabel("Probability of readmission")
        plt.suptitle("Figure 3 recreation: Adjusted readmission rate by diagnosis and HbA1c")
        self.save_plot(figure, "paper_figure3.png")

    def run(self) -> None:
        self.load_data()
        self.recreate_paper_plots_and_export()
    