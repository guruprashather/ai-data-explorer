import sweetviz as sv
import pandas as pd

def generate_sweetviz_report(dataframe, report_file="sweetviz_report.html"):
    report = sv.analyze(dataframe)
    report.show_html(report_file)
    return report_file
