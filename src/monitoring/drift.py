from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def run_drift_report(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("drift_report.html")

    print("📊 Drift report generated")

