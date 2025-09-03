# -*- coding: utf-8 -*-
# =============================================================================
# GC Analysis Pipeline GUI (AttributeError Fix for Session State)
#
# Description:
# This definitive version fixes the critical AttributeError by using the .get()
# method for safer access to session_state, making the app robust against
# unexpected state loss during reruns.
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
if 'sample_definitions' not in st.session_state: st.session_state.sample_definitions = {}
if 'peak_analysis_complete' not in st.session_state: st.session_state.peak_analysis_complete = False

# --- Script Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_ANALYZER_SCRIPT = os.path.join(SCRIPT_DIR, "1_batch_analyzer.py")
CALIBRATION_SCRIPT = os.path.join(SCRIPT_DIR, "2_calibration_analyzer.py")

def assign_analytes_to_peaks(df, analytes_dict):
    """Assigns an analyte name to each peak in the DataFrame."""
    df['Analyte'] = 'Unidentified'
    for name, properties in analytes_dict.items():
        rt_min = properties['rt'] - properties['tolerance']
        rt_max = properties['rt'] + properties['tolerance']
        df.loc[(df['Retention_Time(s)'] >= rt_min) & (df['Retention_Time(s)'] <= rt_max), 'Analyte'] = name
    return df

# ===================================================================
# STEP 1: SET DATA FOLDER
# ===================================================================
st.header("🎯 Step 1: Set Target Data Folder")
path_input = st.text_input("Enter the full path to your chromatogram (.csv) files:", value=st.session_state.data_path)
if path_input != st.session_state.data_path:
    st.session_state.data_path = path_input; st.session_state.peak_analysis_complete = False; st.rerun()
target_path = st.session_state.data_path
if not os.path.isdir(target_path): st.error("Invalid directory path."); st.stop()

# ===================================================================
# STEP 2: DEFINE SAMPLE TYPES
# ===================================================================
st.header("📝 Step 2: Define Sample Types")
try:
    csv_files = sorted([f for f in os.listdir(target_path) if f.endswith('.csv')])
    if not csv_files: st.warning("No CSV files found."); st.stop()
except Exception as e:
    st.error(f"Could not read directory. Error: {e}"); st.stop()

with st.form("sample_definition_form"):
    st.write("For each file, classify it as a Standard, a Sample, or Exclude it from the analysis.")
    temp_definitions = {}
    for csv_file in csv_files:
        st.markdown("---")
        type_map = {"Sample": 0, "Standard": 1, "Exclude": 2}
        default_type = st.session_state.sample_definitions.get(csv_file, {}).get('type', 'Sample')
        default_type_index = type_map.get(default_type, 0)
        cols = st.columns([3, 1]); cols[0].write(f"`{csv_file}`")
        sample_type = cols[1].selectbox("Sample Type", ["Sample", "Standard", "Exclude"], index=default_type_index, key=f"type_{csv_file}")
        temp_definitions[csv_file] = {'type': sample_type}
    if st.form_submit_button("Save Sample Definitions"):
        st.session_state.sample_definitions = temp_definitions; st.success("Sample definitions saved!")

if not st.session_state.sample_definitions: st.info("Please define and save sample types to proceed."); st.stop()

# ===================================================================
# STEP 3: RUN PEAK ANALYSIS
# ===================================================================
st.header("📊 Step 3: Run Peak Analysis")
files_to_process = [fname for fname, props in st.session_state.sample_definitions.items() if props.get('type') != 'Exclude']
if st.button("▶️ Run Peak Analysis", key="run_batch"):
    if not files_to_process: st.error("No files selected for analysis."); st.stop()
    with st.spinner("Analyzing selected chromatograms..."):
        try:
            full_paths = [os.path.join(target_path, f) for f in files_to_process]
            command = ["python", BATCH_ANALYZER_SCRIPT, target_path] + full_paths
            subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            st.session_state.peak_analysis_complete = True; st.success("Peak analysis complete!"); st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Error in '1_batch_analyzer.py'."); st.code(e.stdout); st.code(e.stderr); st.stop()

summary_file = os.path.join(target_path, 'analysis_summary.csv')

# <<< 핵심 수정 사항: .get()을 사용하여 AttributeError를 원천적으로 방지 >>>
# Safely access the session state key. If it doesn't exist, default to False.
if st.session_state.get('peak_analysis_complete', False) and os.path.exists(summary_file):
    st.subheader("Peak Analysis Results"); st.dataframe(pd.read_csv(summary_file))
else:
    st.info("Click 'Run Peak Analysis' to proceed."); st.stop()

# ===================================================================
# STEP 4: DEFINE ANALYTES
# ===================================================================
st.header("🧬 Step 4: Define Analytes of Interest")
with st.form("analyte_form"):
    st.info("Define the analytes you want to quantify in this analysis session.")
    num_analytes = st.number_input("How many analytes?", min_value=1, value=1, step=1)
    analytes_list = []
    for i in range(num_analytes):
        cols = st.columns(3)
        name = cols[0].text_input(f"Analyte #{i+1} Name", key=f"analyte_name_{i}")
        rt = cols[1].number_input(f"Expected RT (s)", key=f"analyte_rt_{i}", format="%.2f")
        tolerance = cols[2].number_input(f"RT Tolerance (±s)", value=2.0, key=f"analyte_tol_{i}", format="%.2f")
        analytes_list.append({'Analyte_Name': name, 'Expected_RT(s)': rt, 'RT_Tolerance(±s)': tolerance})
    submitted_analytes = st.form_submit_button("Save Analyte Definitions")

if not submitted_analytes and 'analytes_list' not in st.session_state:
    st.info("Please define your analytes and save to proceed."); st.stop()
if submitted_analytes:
    st.session_state.analytes_list = analytes_list
    st.success("Analytes defined!")

# ===================================================================
# STEP 5: CALIBRATE AND QUANTIFY
# ===================================================================
st.header("🧪 Step 5: Calibrate and Quantify")
if 'analytes_list' in st.session_state:
    calib_method = st.radio("How to calibrate?", ("Create new curve from this run's Standards", "Load existing curve from file"))
    if calib_method == "Create new curve from this run's Standards":
        with st.form("new_calib_form"):
            st.subheader("Define Standard Concentrations")
            standard_files = [fname for fname, props in st.session_state.sample_definitions.items() if props['type'] == 'Standard']
            if not standard_files: st.warning("No files were classified as 'Standard'.")
            standard_conc_list = []
            for csv_file in standard_files:
                st.markdown(f"**Standard:** `{csv_file}`")
                conc_cols = st.columns(len(st.session_state.analytes_list))
                for i, analyte in enumerate(st.session_state.analytes_list):
                    if analyte['Analyte_Name']:
                        with conc_cols[i]:
                            conc = st.number_input(f"{analyte['Analyte_Name']}", key=f"conc_{csv_file}_{analyte['Analyte_Name']}", min_value=0.0, format="%.4f")
                            standard_conc_list.append({'Filename': csv_file.replace('.csv', ''), 'Analyte_Name': analyte['Analyte_Name'], 'Concentration': conc})
            submitted_new = st.form_submit_button("✅ Run Analysis with New Curve")
        if submitted_new:
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
    else:
        st.subheader("Load Saved Calibration Raw Data")
        uploaded_file = st.file_uploader("Upload `calibration_raw_data.csv`", type=['csv'])
        if st.button("✅ Run Analysis with Loaded Curve"):
            if not uploaded_file: st.error("Please upload a file."); st.stop()
            with st.spinner("Loading curve and analyzing..."):
                try:
                    full_calib_df = pd.read_csv(uploaded_file)
                    user_defined_analyte_names = [a['Analyte_Name'] for a in st.session_state.analytes_list]
                    filtered_calib_df = full_calib_df[full_calib_df['Analyte'].isin(user_defined_analyte_names)]
                    if filtered_calib_df.empty: st.error(f"Uploaded file has no data for: {user_defined_analyte_names}"); st.stop()
                    calibration_models = {}
                    for analyte_name in user_defined_analyte_names:
                        analyte_standards = filtered_calib_df[filtered_calib_df['Analyte'] == analyte_name]
                        if analyte_standards.empty or analyte_standards['Concentration'].nunique() < 2: continue
                        x, y = analyte_standards['Concentration'], analyte_standards['Peak_Area']
                        slope, intercept, _, _, _ = stats.linregress(x, y)
                        calibration_models[analyte_name] = {'slope': slope, 'intercept': intercept}
                    peak_df = pd.read_csv(summary_file); peak_df = peak_df[peak_df['Filename'] != '----------'].copy()
                    peak_df['Filename'] = peak_df['Filename'].replace('', np.nan).ffill()
                    analytes_dict_from_user = {a['Analyte_Name']: {'rt': a['Expected_RT(s)'], 'tolerance': a['RT_Tolerance(±s)']} for a in st.session_state.analytes_list}
                    peak_df = assign_analytes_to_peaks(peak_df, analytes_dict_from_user)
                    sample_files = [fname for fname, props in st.session_state.sample_definitions.items() if props['type'] == 'Sample']
                    unknown_df = peak_df[peak_df['Filename'].isin(sample_files) & (peak_df['Analyte'] != 'Unidentified')].copy()
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

