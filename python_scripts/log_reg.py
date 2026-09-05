from pathlib import Path
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

@dataclass
class logregressionpipeline:
    project_root: Path
    data_dir: Path
    plot_dir: Path

    data: pd.DataFrame | None = field(default=None, init=False)
    model_data: pd.DataFrame | None = field(default=None, init=False)

    train_features: pd.DataFrame | None = field(default=None, init=False)
    test_features: pd.DataFrame | None = field(default=None, init=False)
    train_target: pd.Series | None = field(default=None, init=False)
    test_target: pd.Series | None = field(default=None, init=False)

    model: LogisticRegression | None = field(default=None, init=False)
    predictions: np.ndarray | None = field(default=None, init=False)
    prediction_probabilities: np.ndarray | None = field(default=None, init=False)

    def save_plot(self, fig: plt.Figure, file_name: str, dpi: int = 150):
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.plot_dir / file_name
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def load_data(self):
        fact_table = pd.read_csv(self.data_dir / "fact_encounter.csv")
        diagnosis_table = pd.read_csv(self.data_dir / "dim_diagnosis.csv")
        self.data = fact_table.merge(diagnosis_table, on="diagnosis_key", how="left", validate="many_to_one")

    def prepare_model_data(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data first.")

        patient_level_data = (self.data.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first").copy())
        self.model_data = patient_level_data

        continuous_feature_columns = [
            "time_in_hospital",
            "num_medications",
            "number_inpatient",
            "number_emergency"]

        categorical_feature_columns = [
            "age",
            "medical_specialty",
            "diagnosis_category"]

        model_feature_columns = continuous_feature_columns + categorical_feature_columns
        model_feature_matrix = patient_level_data[model_feature_columns]
        encoded_feature_matrix = pd.get_dummies(model_feature_matrix,columns=categorical_feature_columns,drop_first=True)
        readmission_target = patient_level_data["readmit_30"]

        return encoded_feature_matrix, readmission_target
    
    def train_model(self):
        model_features, readmission_target = self.prepare_model_data()

        (self.train_features, self.test_features, self.train_target, self.test_target) = train_test_split(model_features, readmission_target, test_size=0.2, random_state=42, stratify=readmission_target)
        log_regression_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        log_regression_model.fit(self.train_features, self.train_target)

        self.model = log_regression_model
        self.predictions = log_regression_model.predict(self.test_features)
        self.prediction_probabilities = log_regression_model.predict_proba(self.test_features)[:, 1]

        print(classification_report(self.test_target, self.predictions))
        print("AUC:", roc_auc_score(self.test_target, self.prediction_probabilities))
        print(confusion_matrix(self.test_target, self.predictions))

    def export_reliable_coefficients(self) -> pd.DataFrame:
        if self.model is None or self.train_features is None or self.model_data is None:
            raise ValueError("Model is not trained. Call train_model first.")
    
        minimum_observation_count = 30
        coefficient_table = pd.DataFrame({"feature_name": self.train_features.columns,"coefficient": self.model.coef_[0]})
        coefficient_table["odds_ratio"] = np.exp(coefficient_table["coefficient"])

        category_counts_by_field = {
            "medical_specialty": self.model_data["medical_specialty"].value_counts().to_dict(),
            "diagnosis_category": self.model_data["diagnosis_category"].value_counts().to_dict(),
            "age": self.model_data["age"].value_counts().to_dict(),
        }

        def get_support_count(feature_name: str) -> int:
            for field_name, counts in category_counts_by_field.items():
                prefix = f"{field_name}_"
                if feature_name.startswith(prefix):
                    category_value = feature_name.replace(prefix, "", 1)
                    return int(counts.get(category_value, 0))
            return len(self.model_data)

        coefficient_table["support_count"] = coefficient_table["feature_name"].apply(get_support_count)

        reliable_coefficients = (
            coefficient_table[coefficient_table["support_count"] >= minimum_observation_count]
            .sort_values("odds_ratio", ascending=False).reset_index(drop=True))

        reliable_coefficients.to_csv(self.data_dir / "log_reg_reliable_coefficients.csv", index=False)
        return reliable_coefficients

    def score_full_dataset(self):
        if self.data is None or self.model is None or self.train_features is None:
            raise ValueError("Model is not trained. Call train_model first.")

        continuous_feature_columns = [
            "time_in_hospital",
            "num_medications",
            "number_inpatient",
            "number_emergency",
        ]

        categorical_feature_columns = [
            "age",
            "medical_specialty",
            "diagnosis_category",
        ]

        full_dataset_feature_columns = (continuous_feature_columns + categorical_feature_columns)

        full_dataset_feature_matrix = pd.get_dummies(self.data[full_dataset_feature_columns], 
            columns=categorical_feature_columns, drop_first=True)
        aligned_feature_matrix = full_dataset_feature_matrix.reindex(
            columns=self.train_features.columns, fill_value=0)

        self.data["readmit_risk_score"] = self.model.predict_proba(aligned_feature_matrix)[:, 1]
        risk_score_table = self.data[["encounter_id", "readmit_risk_score"]].copy()

        risk_score_table.to_csv(self.data_dir / "risk_scores.csv", index=False)
        
    def make_core_model_plots(self):
        if self.test_target is None or self.predictions is None or self.prediction_probabilities is None:
            raise ValueError("Prediction outputs are missing. Call train_model first.")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            self.test_target, 
            self.prediction_probabilities)

        auc_value = roc_auc_score(
            self.test_target, 
            self.prediction_probabilities)

        roc_figure = plt.figure(figsize=(6, 6))
        plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"Model (AUC = {auc_value:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guessing (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - 30-day readmission model")
        plt.legend()
        self.save_plot(roc_figure, "log_reg_roc_curve.png")

        risk_score_frame = pd.DataFrame(
            {"risk_score": self.prediction_probabilities,
            "actual_outcome": self.test_target.map(
            {0: "Not readmitted", 1: "Readmitted <30 days"}).to_numpy()})

        risk_distribution_figure = plt.figure(figsize=(8, 5))
        sns.histplot(data=risk_score_frame, x="risk_score", hue="actual_outcome", bins=30, stat="density", common_norm=False)
        plt.xlabel("Predicted readmission probability")
        plt.ylabel("Density")
        plt.title("Predicted risk score distribution by actual outcome")
        self.save_plot(risk_distribution_figure, "log_reg_risk_distribution.png")

        confusion_matrix_values = confusion_matrix(self.test_target, self.predictions)
        confusion_matrix_figure = plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix_values, annot=True, fmt="d", cmap="Blues", xticklabels=["Not readmitted", "Readmitted"], yticklabels=["Not readmitted", "Readmitted"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        self.save_plot(confusion_matrix_figure, "log_reg_confusion_matrix.png")
        
    def export_dim_admission_source(self):
        admission_source_mapping = pd.read_csv(self.data_dir / "IDS_mapping.csv", skiprows=42, skip_blank_lines=True, keep_default_na=False)
        admission_source_mapping["admission_source_id"] = (admission_source_mapping["admission_source_id"].astype(int))
        admission_source_mapping["description"] = (admission_source_mapping["description"].str.strip())
        admission_source_mapping.to_csv(self.data_dir / "dim_admission_source.csv", index=False)

    def run(self):
        self.load_data()
        self.train_model()
        self.export_reliable_coefficients()
        self.score_full_dataset()
        self.make_core_model_plots()
        self.export_dim_admission_source()