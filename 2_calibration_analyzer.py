# -*- coding: utf-8 -*-
# =============================================================================
# Excel-Based Calibration and Quantitation Analyzer (Single File Export)
#
# Description:
# This version is updated to consolidate all raw calibration data into a
# single output file, `calibration_raw_data.csv`, instead of multiple
# files per analyte, as requested by the user for simpler management.
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def assign_analytes_to_peaks(df, analytes_dict):
    """Assigns an analyte name to each peak based on its RT."""
    df['Analyte'] = 'Unidentified'
    for name, properties in analytes_dict.items():
        rt_min = properties['rt'] - properties['tolerance']
        rt_max = properties['rt'] + properties['tolerance']
        df.loc[(df['Retention_Time(s)'] >= rt_min) & (df['Retention_Time(s)'] <= rt_max), 'Analyte'] = name
    return df


def main():
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = '.'
    print(f"--- Starting Calibration in: {target_path} ---")

    summary_file = os.path.join(target_path, 'analysis_summary.csv')
    parameter_file = os.path.join(target_path, 'analysis_parameters.xlsx')

    try:
        peak_df = pd.read_csv(summary_file);
        peak_df = peak_df[peak_df['Filename'] != '----------'].copy()
        peak_df['Filename'] = peak_df['Filename'].replace('', np.nan).ffill()
        peak_df['Retention_Time(s)'] = pd.to_numeric(peak_df['Retention_Time(s)'])
        peak_df['Peak_Area'] = pd.to_numeric(peak_df['Peak_Area'])
        analyte_definitions = pd.read_excel(parameter_file, sheet_name='Analyte_Definitions')
        standard_concentrations = pd.read_excel(parameter_file, sheet_name='Standard_Concentrations')
    except Exception as e:
        print(f"Error reading files: {e}");
        sys.exit(1)

    peak_df['Base_Filename'] = peak_df['Filename'].str.replace('.csv', '', regex=False)
    standard_concentrations.rename(columns={'Filename': 'Base_Filename', 'Analyte_Name': 'Analyte'}, inplace=True)
    analytes_dict = {row['Analyte_Name']: {'rt': row['Expected_RT(s)'], 'tolerance': row['RT_Tolerance(±s)']} for _, row
                     in analyte_definitions.iterrows()}
    peak_df = assign_analytes_to_peaks(peak_df, analytes_dict)
    standards_df = pd.merge(standard_concentrations, peak_df, on=['Base_Filename', 'Analyte'], how='left').dropna(
        subset=['Peak_Area'])

    calibration_models = {}
    for analyte_name in analyte_definitions['Analyte_Name']:
        analyte_standards = standards_df[standards_df['Analyte'] == analyte_name]
        if analyte_standards['Concentration'].nunique() < 2:
            print(f"Warning: Not enough unique concentration points for '{analyte_name}'. Skipping.")
            continue

        x, y = analyte_standards['Concentration'], analyte_standards['Peak_Area']
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        calibration_models[analyte_name] = {'slope': slope, 'intercept': intercept}
        print(f"\nAnalyte: {analyte_name}, R-squared: {r_value ** 2:.4f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label='Standard Data');
        plt.plot(x, slope * x + intercept, 'r-', label='Linear Fit')
        plt.title(f'Calibration Curve for {analyte_name}');
        plt.xlabel('Concentration');
        plt.ylabel('Peak Area');
        plt.legend();
        plt.grid(True)
        plt.text(0.05, 0.95, f'y = {slope:.4g}x + {intercept:.4g}\n$R^2$ = {r_value ** 2:.4f}',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plot_filename = os.path.join(target_path, f'calibration_curve_{analyte_name}.png')
        plt.savefig(plot_filename);
        plt.close()
        print(f"  - Plot saved to '{os.path.basename(plot_filename)}'")

    # <<< 핵심 수정 사항: 모든 표준 시료 데이터를 하나의 파일로 저장 >>>
    if not standards_df.empty:
        raw_data_filename = os.path.join(target_path, 'calibration_raw_data.csv')
        columns_to_save = ['Filename', 'Analyte', 'Concentration', 'Peak_Area', 'Retention_Time(s)']
        existing_columns = [col for col in columns_to_save if col in standards_df.columns]
        standards_df[existing_columns].to_csv(raw_data_filename, index=False)
        print(f"  - All raw calibration data saved to '{os.path.basename(raw_data_filename)}'")

    standard_base_filenames = standard_concentrations['Base_Filename'].unique()
    unknown_df = peak_df[
        ~peak_df['Base_Filename'].isin(standard_base_filenames) & (peak_df['Analyte'] != 'Unidentified')].copy()

    def calculate_conc(row):
        model = calibration_models.get(row['Analyte'])
        if model and model['slope'] != 0: return (row['Peak_Area'] - model['intercept']) / model['slope']
        return np.nan

    unknown_df['Calculated_Concentration'] = unknown_df.apply(calculate_conc, axis=1)

    final_report_df = unknown_df[['Filename', 'Analyte', 'Calculated_Concentration']].dropna()
    if not final_report_df.empty:
        report_filename = os.path.join(target_path, 'final_concentration_report.csv')
        final_report_df.to_csv(report_filename, index=False)
        print(f"\nQuantitation Complete. Report saved to '{os.path.basename(report_filename)}'")
        print(final_report_df.to_string())


if __name__ == '__main__':
    main()

