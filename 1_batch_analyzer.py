# -*- coding: utf-8 -*-
# =============================================================================
# Batch Chromatogram Analyzer (Explicit File List Version)
#
# Description:
# This version is updated to accept an explicit list of file paths to
# process as command-line arguments. If no list is provided, it falls
# back to finding all CSVs in the target directory for standalone use.
# =============================================================================

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

try:
    from pybaselines.whittaker import arpls
except ImportError:
    print("pybaselines library not found.")
    exit()

# Settings and core functions remain the same
PEAK_DETECTION_SETTINGS = {'height': 0.02, 'distance': 50, 'prominence': 0.02}
BASELINE_SETTINGS = {'lam': 1e7}


def process_chromatogram_file(filepath, peak_settings, baseline_settings):
    try:
        df = pd.read_csv(filepath)
        time, voltage = df['Time (s)'].values, df['Voltage (V)'].values
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

    baseline, _ = arpls(voltage, **baseline_settings)
    corrected = voltage - baseline
    indices, props = find_peaks(corrected, **peak_settings)

    if indices.size == 0: return {'filename': os.path.basename(filepath), 'data': df, 'peaks_found': False}

    results = {'filename': os.path.basename(filepath), 'retention_times': [], 'peak_areas': [], 'peak_indices': indices,
               'data': df, 'original_voltage': voltage, 'calculated_baseline': baseline, 'voltage_corrected': corrected,
               'peaks_found': True, 'peak_properties': props}
    for i, idx in enumerate(indices):
        l, r = props['left_bases'][i], props['right_bases'][i]
        area = trapezoid(corrected[l:r + 1], time[l:r + 1])
        results['retention_times'].append(time[idx])
        results['peak_areas'].append(area)
    return results


def create_and_save_plot(results, target_path):
    if not results or not results['peaks_found']: return
    filename = results['filename']
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(results['data']['Time (s)'], results['original_voltage'], color='lightgray', label='Original Signal')
    ax.plot(results['data']['Time (s)'], results['calculated_baseline'], '--', color='red',
            label='Calculated Baseline (ALS)')
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['peak_indices'])))
    for i, idx in enumerate(results['peak_indices']):
        props, rt = results['peak_properties'], results['retention_times'][i]
        height = results['voltage_corrected'][idx] + results['calculated_baseline'][idx]
        ax.plot(rt, height, 'o', color=colors[i], label=f'Peak {i + 1} (RT: {rt:.2f} s)')
        l, r = props['left_bases'][i], props['right_bases'][i]
        time_slice = results['data']['Time (s)'][l:r + 1]
        ax.fill_between(time_slice, results['original_voltage'][l:r + 1], results['calculated_baseline'][l:r + 1],
                        where=(results['original_voltage'][l:r + 1] > results['calculated_baseline'][l:r + 1]),
                        color=colors[i], alpha=0.5)
    ax.set_title(f'Baseline Correction: {filename}');
    ax.set_xlabel('Time (s)');
    ax.set_ylabel('Signal (Voltage)');
    ax.legend();
    ax.grid(True)
    output_filename = os.path.join(target_path, filename.replace('.csv', '_peaks.png'))
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 1_batch_analyzer.py <target_path> [file1, file2, ...]")
        return

    target_path = sys.argv[1]

    # <<< 핵심 수정 사항: 외부에서 파일 목록을 전달받아 사용 >>>
    if len(sys.argv) > 2:
        # If a list of files is provided by the GUI, use it directly
        csv_files_to_process = sys.argv[2:]
        print(f"--- Starting Batch Analysis on {len(csv_files_to_process)} specified files in: {target_path} ---")
    else:
        # Fallback for standalone use: find all CSVs and exclude outputs
        print(f"--- Starting Batch Analysis on all CSVs in: {target_path} ---")
        all_csv_files = glob.glob(os.path.join(target_path, '*.csv'))
        output_files = ['analysis_summary.csv', 'final_concentration_report.csv']
        output_paths = [os.path.join(target_path, f) for f in output_files]
        csv_files_to_process = [f for f in all_csv_files if f not in output_paths]

    if not csv_files_to_process:
        print("No valid data CSV files found to process.")
        return

    summary_data = []
    # Sort files by name before processing
    sorted_files = sorted(csv_files_to_process,
                          key=lambda f: (0, os.path.basename(f)) if 'std' in os.path.basename(f).lower() else (
                          1, os.path.basename(f)))

    for i, filepath in enumerate(sorted_files):
        print(f"Processing '{os.path.basename(filepath)}'...")
        results = process_chromatogram_file(filepath, PEAK_DETECTION_SETTINGS, BASELINE_SETTINGS)
        if results and results['peaks_found']:
            create_and_save_plot(results, target_path)
            for j, rt in enumerate(results['retention_times']):
                summary_data.append({'Filename': results['filename'] if j == 0 else '', 'Retention_Time(s)': rt,
                                     'Peak_Area': results['peak_areas'][j]})
            if i < len(sorted_files) - 1:
                summary_data.append(
                    {'Filename': '----------', 'Retention_Time(s)': '----------', 'Peak_Area': '----------'})

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_filename = os.path.join(target_path, 'analysis_summary.csv')
        summary_df.to_csv(summary_filename, index=False)
        print(f"Batch analysis complete. Summary saved to '{summary_filename}'.")


if __name__ == '__main__':
    main()

