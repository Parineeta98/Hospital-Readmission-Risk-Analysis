from __future__ import annotations
from pathlib import Path
from eda import eda
from log_reg import logregressionpipeline
from recreate_plots_from_strack import recreateplotsfromstrack

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    result_plot_dir = project_root / "result_plot"

    run_eda = input("Run EDA file? [yes/no]: ").strip().lower() == "yes"
    run_logreg = input("Run logistic regression file? [yes/no]: ").strip().lower() == "yes"
    run_paper_plots = input("Replicate plots from Strack et al.? [yes/no]: ").strip().lower() == "yes"

    if run_eda:
        eda_pipeline = eda(project_root=project_root, data_dir=data_dir, plot_dir=result_plot_dir)
        eda_pipeline.run()

    if run_logreg:
        log_reg_pipeline = logregressionpipeline(project_root=project_root, data_dir=data_dir, plot_dir=result_plot_dir)
        log_reg_pipeline.run()

    if run_paper_plots:
        paper_pipeline = recreateplotsfromstrack(project_root=project_root, data_dir=data_dir, plot_dir=result_plot_dir)
        paper_pipeline.run()

if __name__ == "__main__":
    main()
    print("All done. Plots saved to: result_plot")