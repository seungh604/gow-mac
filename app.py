# -*- coding: utf-8 -*-
# =============================================================================
# GC Analysis Pipeline GUI (Exclude Files Logic)
#
# Description:
# This definitive version inverts the file selection logic. The user now
# checks the files they wish to EXCLUDE from the analysis, which is more
# efficient when working with large numbers of files.
# =============================================================================

import streamlit as st
import pandas as pd
import subprocess
import os
import glob
import numpy as np
from scipy import stats
from zipfile import ZipFile

# --- App Configuration & Initialization ---
st.set_page_config(page_title="GC Analysis Pipeline", page_icon="🔬", layout="wide")
st.title("🔬 GOW-MAC GC Analysis Pipeline")

# Initialize session state variables
if 'data_path' not in st.session_state: st.session_state.data_path = os.getcwd()
if 'excluded_files' not in st.session_state: st.session_state.excluded_files = []
if 'sample_definitions' not in st.session_state: st.session_state.sample_definitions = {}
if 'peak_analysis_complete' not in st.session_state: st.session_state.peak_analysis_complete = False

# --- Script Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_ANALYZER_SCRIPT = os.path.join(SCRIPT_DIR, "1_batch_analyzer.py")
CALIBRATION_SCRIPT = os.path.join(SCRIPT_DIR, "2_calibration_analyzer.py")

def assign_analytes_to_peaks(df, analytes_dict):
    df['Analyte'] = 'Unidentified'
    for name, properties in analytes_dict.items():
        rt_min, rt_max = properties['rt'] - properties['tolerance'], properties['rt'] + properties['tolerance']
        df.loc[(df['Retention_Time(s)'] >= rt_min) & (df['Retention_Time(s)'] <= rt_max), 'Analyte'] = name
    return df

# ===================================================================
# STEP 1: SET DATA FOLDER
# ===================================================================
st.header("🎯 Step 1: Set Target Data Folder")
path_input = st.text_input("Enter the full path to your chromatogram (.csv) files:", value=st.session_state.data_path)
if path_input != st.session_state.data_path:
    st.session_state.data_path = path_input
    st.session_state.excluded_files = []
    st.session_state.sample_definitions = {}
    st.session_state.peak_analysis_complete = False
    st.rerun()
target_path = st.session_state.data_path
if not os.path.isdir(target_path): st.error("Invalid directory path."); st.stop()

# ===================================================================
# STEP 2: EXCLUDE FILES FROM ANALYSIS
# ===================================================================
st.header("☑️ Step 2: Exclude Files from Analysis")
try:
    all_csv_files = sorted([f for f in os.listdir(target_path) if f.endswith('.csv')])
    if not all_csv_files: st.warning("No CSV files found in the specified directory."); st.stop()
except Exception as e:
    st.error(f"Could not read directory. Error: {e}"); st.stop()

with st.form("file_exclusion_form"):
    st.write("Check the files you want to **EXCLUDE** from this analysis run.")
    temp_excluded_files = []
    for csv_file in all_csv_files:
        is_checked = st.checkbox(f"Exclude `{csv_file}`", value=(csv_file in st.session_state.excluded_files), key=f"exclude_{csv_file}")
        if is_checked:
            temp_excluded_files.append(csv_file)
    if st.form_submit_button("Confirm Excluded Files"):
        st.session_state.excluded_files = temp_excluded_files
        st.success(f"{len(st.session_state.excluded_files)} files have been excluded.")

# Define the list of files that will actually be processed
files_to_include = [f for f in all_csv_files if f not in st.session_state.excluded_files]

if not files_to_include:
    st.warning("All files have been excluded. Please uncheck at least one file to proceed."); st.stop()

# ===================================================================
# STEP 3: DEFINE SAMPLE TYPES for Included Files
# ===================================================================
st.header("📝 Step 3: Define Sample Types for Included Files")
with st.form("sample_definition_form"):
    st.write("For each of the files you have included, classify it as a Standard or a Sample.")
    temp_definitions = {}
    for csv_file in files_to_include: # Only loop through included files
        st.markdown("---")
        type_map = {"Sample": 0, "Standard": 1}
        default_type = st.session_state.sample_definitions.get(csv_file, {}).get('type', 'Sample')
        default_type_index = type_map.get(default_type, 0)
        cols = st.columns([3, 1]); cols[0].write(f"`{csv_file}`")
        sample_type = cols[1].selectbox("Sample Type", ["Sample", "Standard"], index=default_type_index, key=f"type_{csv_file}")
        temp_definitions[csv_file] = {'type': sample_type}
    if st.form_submit_button("Save Sample Definitions"):
        st.session_state.sample_definitions = temp_definitions; st.success("Sample definitions saved!")

if not st.session_state.sample_definitions: st.info("Please define and save sample types to proceed."); st.stop()

# ===================================================================
# STEP 4: RUN PEAK ANALYSIS on Included Files
# ===================================================================
st.header("📊 Step 4: Run Peak Analysis")
st.info(f"**{len(files_to_include)} files** will be analyzed.")

if st.button("▶️ Run Peak Analysis", key="run_batch"):
    with st.spinner("Analyzing selected chromatograms..."):
        try:
            full_paths = [os.path.join(target_path, f) for f in files_to_include]
            command = ["python", BATCH_ANALYZER_SCRIPT, target_path] + full_paths
            subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            st.session_state.peak_analysis_complete = True; st.success("Peak analysis complete!"); st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Error in '1_batch_analyzer.py'."); st.code(e.stdout); st.code(e.stderr); st.stop()

summary_file = os.path.join(target_path, 'analysis_summary.csv')
if not st.session_state.peak_analysis_complete or not os.path.exists(summary_file):
    st.info("Click 'Run Peak Analysis' to proceed."); st.stop()
st.subheader("Peak Analysis Results"); st.dataframe(pd.read_csv(summary_file))

# ===================================================================
# STEP 5 & 6: CALIBRATE AND QUANTIFY
# ===================================================================
st.header("🧪 Step 5 & 6: Calibrate and Quantify")
if 'analytes_list' not in st.session_state: st.session_state.analytes_list = []
# The rest of the logic remains unchanged as it works based on the generated summary and definitions.
calib_method = st.radio("How to calibrate?", ("Create new curve from this run's Standards", "Load existing curve from file"))
if calib_method == "Create new curve from this run's Standards":
    with st.form("new_calib_form"):
        st.subheader("Define Analytes")
        num_analytes = st.number_input("How many analytes?", min_value=1, value=max(1, len(st.session_state.analytes_list)), step=1)
        analytes_list = []
        for i in range(num_analytes):
            cols = st.columns(3)
            prev_name = st.session_state.analytes_list[i]['Analyte_Name'] if i < len(st.session_state.analytes_list) else ""
            name = cols[0].text_input(f"Analyte #{i+1} Name", value=prev_name, key=f"analyte_name_{i}")
            rt = cols[1].number_input(f"Expected RT (s)", key=f"analyte_rt_{i}", format="%.2f")
            tolerance = cols[2].number_input(f"RT Tolerance (±s)", value=2.0, key=f"analyte_tol_{i}", format="%.2f")
            analytes_list.append({'Analyte_Name': name, 'Expected_RT(s)': rt, 'RT_Tolerance(±s)': tolerance})
        st.subheader("Define Standard Concentrations")
        standard_files = [fname for fname, props in st.session_state.sample_definitions.items() if props['type'] == 'Standard']
        if not standard_files: st.warning("No files were classified as 'Standard'.")
        standard_conc_list = []
        for csv_file in standard_files:
            st.markdown(f"**Standard:** `{csv_file}`")
            conc_cols = st.columns(len(analytes_list))
            for i, analyte in enumerate(analytes_list):
                if analyte['Analyte_Name']:
                    with conc_cols[i]:
                        conc = st.number_input(f"{analyte['Analyte_Name']}", key=f"conc_{csv_file}_{analyte['Analyte_Name']}", min_value=0.0, format="%.4f")
                        standard_conc_list.append({'Filename': csv_file.replace('.csv', ''), 'Analyte_Name': analyte['Analyte_Name'], 'Concentration': conc})
        submitted_new = st.form_submit_button("✅ Run Analysis with New Curve")
    if submitted_new:
        st.session_state.analytes_list = analytes_list
        analyte_definitions_df = pd.DataFrame(st.session_state.analytes_list); standard_concentrations_df = pd.DataFrame(standard_conc_list)
        if analyte_definitions_df.empty or analyte_definitions_df['Analyte_Name'].str.strip().eq('').any(): st.error("Please define analyte names."); st.stop()
        if standard_concentrations_df.empty: st.error("Please provide standard concentrations."); st.stop()
        can_calibrate = any(standard_concentrations_df[standard_concentrations_df['Analyte_Name'] == name]['Concentration'].nunique() >= 2 for name in analyte_definitions_df['Analyte_Name'])
        if not can_calibrate: st.error("At least one analyte must have >= 2 different concentration levels."); st.stop()
        with st.spinner("Generating new calibration..."):
            param_file = os.path.join(target_path, 'analysis_parameters.xlsx')
            with pd.ExcelWriter(param_file, engine='openpyxl') as writer:
                analyte_definitions_df.to_excel(writer, sheet_name='Analyte_Definitions', index=False)
                standard_concentrations_df.to_excel(writer, sheet_name='Standard_Concentrations', index=False)
            try:
                subprocess.run(["python", CALIBRATION_SCRIPT, target_path], check=True, capture_output=True, text=True, encoding='utf-8')
                st.success("Analysis complete!")
            except subprocess.CalledProcessError as e:
                st.error(f"Error in '2_calibration_analyzer.py'."); st.code(e.stdout); st.code(e.stderr); st.stop()
            report_file = os.path.join(target_path, 'final_concentration_report.csv')
            if os.path.exists(report_file): st.subheader("Final Report"); st.dataframe(pd.read_csv(report_file))
            curve_files = sorted(glob.glob(os.path.join(target_path, 'calibration_curve_*.png')))
            if curve_files: st.subheader("Generated Curves"); [st.image(f) for f in curve_files]
else: # Mode B: Load Existing Curve
    st.subheader("Load Saved Calibration Raw Data")
    uploaded_file = st.file_uploader("Upload `calibration_raw_data.csv`", type=['csv'])
    if st.button("✅ Run Analysis with Loaded Curve"):
        if not uploaded_file: st.error("Please upload a file."); st.stop()
        with st.spinner("Loading curve and analyzing..."):
            try:
                full_calib_df = pd.read_csv(uploaded_file)
                peak_df = pd.read_csv(summary_file); peak_df = peak_df[peak_df['Filename'] != '----------'].copy()
                peak_df['Filename'] = peak_df['Filename'].replace('', np.nan).ffill()
                analytes_dict_from_calib = {name: {'rt': group['Retention_Time(s)'].mean(), 'tolerance': 2.0} for name, group in full_calib_df.groupby('Analyte')}
                peak_df = assign_analytes_to_peaks(peak_df, analytes_dict_from_calib)
                sample_files = [fname for fname, props in st.session_state.sample_definitions.items() if props['type'] == 'Sample']
                unknown_df = peak_df[peak_df['Filename'].isin(sample_files) & (peak_df['Analyte'] != 'Unidentified')].copy()
                calibration_models = {}
                for analyte_name in full_calib_df['Analyte'].unique():
                    analyte_standards = full_calib_df[full_calib_df['Analyte'] == analyte_name]
                    if analyte_standards.empty or analyte_standards['Concentration'].nunique() < 2: continue
                    x, y = analyte_standards['Concentration'], analyte_standards['Peak_Area']
                    slope, intercept, _, _, _ = stats.linregress(x, y)
                    calibration_models[analyte_name] = {'slope': slope, 'intercept': intercept}
                def calculate_conc(row):
                    model = calibration_models.get(row['Analyte'])
                    if model and model['slope'] != 0: return (row['Peak_Area'] - model['intercept']) / model['slope']
                    return np.nan
                unknown_df['Calculated_Concentration'] = unknown_df.apply(calculate_conc, axis=1)
                final_report_df = unknown_df[['Filename', 'Analyte', 'Calculated_Concentration']].dropna()
                if not final_report_df.empty:
                    st.success("Analysis complete!")
                    st.header("📈 Final Analysis Results"); st.subheader("Final Report"); st.dataframe(final_report_df)
                    report_filename = os.path.join(target_path, 'final_concentration_report.csv')
                    final_report_df.to_csv(report_filename, index=False)
                else:
                    st.warning("No quantifiable peaks found in the 'Sample' files.")
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

