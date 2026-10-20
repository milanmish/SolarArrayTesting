import io
import os
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def convert_central_to_utc(central_time_str):
    """
    Converts local Central Time (CST/CDT) string to UTC datetime string[cite: 5].
    Handles daylight saving time (CDT vs CST) automatically.
    """
    dt_local = pd.to_datetime(central_time_str).tz_localize("America/Chicago")
    dt_utc = dt_local.tz_convert("UTC")
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")

def fetch_noaa_solrad_irradiance(test_datetime_utc, local_file_path=None):
    """
    Downloads NOAA SOLRAD daily file for Madison, WI (msn) and extracts
    the global downwelling irradiance (dw_psp) for the given UTC timestamp[cite: 1, 2, 6].
    Automatically checks QC archive first, then falls back to real-time folder[cite: 1].
    """
    dt = pd.to_datetime(test_datetime_utc)
    year = dt.year
    yy = dt.strftime('%y')
    jjj = dt.strftime('%j')
    filename = f"msn{yy}{jjj}.dat"
    
    content = None
    
    # Read from local file if provided and exists
    if local_file_path and os.path.exists(local_file_path):
        print(f"Reading from local file: {local_file_path}")
        with open(local_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        # Check NOAA QC Archive, then Realtime folder[cite: 1]
        urls = [
            f"https://gml.noaa.gov/aftp/data/radiation/solrad/msn/{year}/{filename}",
            f"https://gml.noaa.gov/aftp/data/radiation/solrad/realtime/{filename}"
        ]
        
        for url in urls:
            try:
                print(f"Attempting to fetch NOAA data: {url}")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    content = response.read().decode('utf-8')
                    print(f"Successfully retrieved SOLRAD file: {filename}\n")
                    break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"  -> File not found at archive path. Checking real-time folder...[cite: 1]")
                    continue
                else:
                    raise e
            except Exception as e:
                raise RuntimeError(f"Connection error: {e}")
                
        if content is None:
            raise FileNotFoundError(
                f"Could not locate {filename} on NOAA GML server[cite: 1].\n"
                f"Ensure the timestamp ({test_datetime_utc} UTC) is valid and data has been published.\n"
                f"You can also pass a manual irradiance value or local `.dat` file path."
            )

    # Post-June 18, 2009 Madison schema (31 columns)[cite: 4, 5]
    columns = [
        'year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen',
        'dw_psp', 'qc_dwpsp', 'direct', 'qc_direct', 'diffuse', 'qc_diffuse',
        'uvb', 'qc_uvb', 'uvb_temp', 'qc_uvb_temp',
        'dpir', 'qc_dpir', 'dpirc', 'qc_dpirc', 'dpird', 'qc_dpird',
        'std_dwpsp', 'std_direct', 'std_diffuse', 'std_uvb', 
        'std_dpir', 'std_dpirc', 'std_dpird'
    ]
    
    # Skip 2 header lines[cite: 5]
    df = pd.read_csv(io.StringIO(content), skiprows=2, sep=r'\s+', names=columns)
    df.replace(-9999.9, np.nan, inplace=True)  # Replace missing value indicators[cite: 2]
    
    # Create UTC datetime index[cite: 5]
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour', 'min']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H-%M'
    )
    df.set_index('datetime', inplace=True)
    
    # Find nearest timestamp record
    nearest_idx = df.index.get_indexer([dt], method='nearest')[0]
    matched_row = df.iloc[nearest_idx]
    
    qc_flag = matched_row['qc_dwpsp']
    g_meas = float(matched_row['dw_psp'])
    
    if qc_flag != 0:
        print(f"Warning: SOLRAD QC flag is {int(qc_flag)} (Non-zero indicates questionable data)[cite: 2].")
        
    print(f"Matched Record Time (UTC): {df.index[nearest_idx]}")
    print(f"NOAA Irradiance (dw_psp): {g_meas:.2f} W/m²\n")
    return g_meas

def extract_voltage_current(df):
    """
    Flexibly detects Voltage and Current columns in the DataFrame,
    stripping whitespace and falling back to 1st/2nd columns if needed.
    """
    df.columns = [str(col).strip() for col in df.columns]
    
    v_col, i_col = None, None
    
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in ['voltage', 'v', 'volts', 'voltage (v)', 'voltage_v', 'v_meas'] or 'volt' in c_lower:
            v_col = col
            break
            
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in ['current', 'i', 'amps', 'current (a)', 'current_a', 'i_meas'] or 'curr' in c_lower or 'amp' in c_lower:
            i_col = col
            break
            
    if v_col is None or i_col is None:
        if len(df.columns) >= 2:
            v_col = df.columns[0]
            i_col = df.columns[1]
            print(f"Auto-selected columns by position -> Voltage: '{v_col}', Current: '{i_col}'")
        else:
            raise KeyError(f"Could not identify Voltage and Current columns. CSV Columns found: {list(df.columns)}")
    else:
        print(f"Detected columns -> Voltage: '{v_col}', Current: '{i_col}'")
        
    return df[v_col].astype(float).values, df[i_col].astype(float).values

def analyze_iv_curve(v, i, num_dense_points=2000):
    """
    Uses Monotonic PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    to interpolate the I-V curve without artificial overshoots, accurately
    extracting Pmp, Vmp, Imp, Voc, Isc, and Fill Factor.
    """
    sort_idx = np.argsort(v)
    v_s, i_s = v[sort_idx], i[sort_idx]
    
    # Remove duplicate voltage points if present
    v_unique, unique_idx = np.unique(v_s, return_index=True)
    i_unique = i_s[unique_idx]
    
    # Monotonic PCHIP Hermite Spline prevents overshoots near Voc
    pchip = PchipInterpolator(v_unique, i_unique)
    v_dense = np.linspace(v_unique.min(), v_unique.max(), num_dense_points)
    i_dense = pchip(v_dense)
    i_dense = np.clip(i_dense, 0, None)  # Ensure non-negative current
    
    p_dense = v_dense * i_dense
    
    idx_mpp = np.argmax(p_dense)
    p_mp = p_dense[idx_mpp]
    v_mp = v_dense[idx_mpp]
    i_mp = i_dense[idx_mpp]
    
    v_oc = v_unique.max()
    i_sc = i_unique.max()
    ff = p_mp / (v_oc * i_sc) if (v_oc * i_sc) > 0 else 0.0
    
    return {
        'Pmp': p_mp, 'Vmp': v_mp, 'Imp': i_mp,
        'Voc': v_oc, 'Isc': i_sc, 'FF': ff
    }

def process_iv_normalization(
    input_csv="XXR Module 1 IV.csv",
    output_csv="iv_stc_normalized.csv",
    test_datetime_central=None,
    local_solrad_file=None,
    G_meas=None,
    T_meas=32.0,
    alpha_rel=0.00058,  # +0.058 %/°C for Maxeon Gen 7
    beta_rel=-0.00236,  # -0.236 %/°C for Maxeon Gen 7
    Rs=0.0              # Series resistance assumed 0
):
    # 1. Read input CSV with flexible delimiter handling
    try:
        df_raw = pd.read_csv(input_csv)
        if len(df_raw.columns) == 1:
            df_raw = pd.read_csv(input_csv, sep=';')
    except Exception as e:
        raise RuntimeError(f"Could not read input CSV file '{input_csv}'. Error: {e}")

    # Extract voltage and current values
    v_meas, i_meas = extract_voltage_current(df_raw)
    p_meas = v_meas * i_meas

    # 2. Obtain irradiance (G_meas)
    if G_meas is None:
        if test_datetime_central is None:
            raise ValueError("Must provide either `test_datetime_central` or explicit `G_meas`.")
        
        # Convert local Central Time (CST/CDT) to UTC[cite: 5]
        test_datetime_utc = convert_central_to_utc(test_datetime_central)
        print(f"Local Central Time : {test_datetime_central}")
        print(f"Converted UTC Time : {test_datetime_utc}")
        
        # Auto-fetch NOAA SOLRAD irradiance
        G_meas = fetch_noaa_solrad_irradiance(
            test_datetime_utc=test_datetime_utc, 
            local_file_path=local_solrad_file
        )
    else:
        print(f"Using manually specified irradiance: {G_meas:.2f} W/m²\n")

    # 3. Analyze measured parameters
    meas_metrics = analyze_iv_curve(v_meas, i_meas)
    Isc1 = meas_metrics['Isc']
    Voc1 = meas_metrics['Voc']
    
    # 4. Temperature coefficients scaling for 24-cell series string
    alpha = alpha_rel * Isc1  # A/°C
    beta = beta_rel * Voc1    # V/°C
    
    # 5. IEC 60891 STC Normalization (G_stc = 1000 W/m^2, T_stc = 25 °C)
    G_stc, T_stc = 1000.0, 25.0
    dT = T_stc - T_meas
    
    i_stc = i_meas + Isc1 * ((G_stc / G_meas) - 1.0) + alpha * dT
    v_stc = v_meas - Rs * (i_stc - i_meas) + beta * dT
    p_stc = v_stc * i_stc
    
    # 6. Analyze STC parameters
    stc_metrics = analyze_iv_curve(v_stc, i_stc)
    
    # 7. Build Output DataFrame
    df_output = pd.DataFrame({
        'V_Measured_V': v_meas,
        'I_Measured_A': i_meas,
        'P_Measured_W': p_meas,
        'V_STC_Point_V': v_stc,
        'I_STC_Point_A': i_stc,
        'P_STC_Point_W': p_stc,
    })
    
    # 8. Construct Summary Header Block for CSV
    summary_header = [
        "SUMMARY METRIC REPORT",
        "Parameter,Measured,STC_Normalized",
        f"Irradiance (W/m²),{G_meas:.2f},{G_stc:.2f}",
        f"Temperature (°C),{T_meas:.1f},{T_stc:.1f}",
        f"Pmp (W),{meas_metrics['Pmp']:.3f},{stc_metrics['Pmp']:.3f}",
        f"Vmp (V),{meas_metrics['Vmp']:.3f},{stc_metrics['Vmp']:.3f}",
        f"Imp (A),{meas_metrics['Imp']:.3f},{stc_metrics['Imp']:.3f}",
        f"Voc (V),{meas_metrics['Voc']:.3f},{stc_metrics['Voc']:.3f}",
        f"Isc (A),{meas_metrics['Isc']:.3f},{stc_metrics['Isc']:.3f}",
        f"Fill Factor,{meas_metrics['FF']:.3f},{stc_metrics['FF']:.3f}",
        "",
        "I-V CURVE POINT DATA"
    ]
    
    header_str = "\n".join(summary_header) + "\n"
    csv_body = df_output.to_csv(index=False)
    
    # Write summary header + point data to output CSV
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write(header_str + csv_body)
    
    # 9. Terminal Summary Report
    print("=" * 58)
    print("                I-V NORMALIZATION RESULTS                 ")
    print("=" * 58)
    print(f"Test Conditions : {G_meas:.2f} W/m² | {T_meas:.1f} °C | 24 Cells")
    print(f"STC Target      : {G_stc:.1f} W/m² | {T_stc:.1f} °C")
    print("-" * 58)
    print(f"{'Metric':<12} | {'Measured':<18} | {'STC Normalized':<18}")
    print("-" * 58)
    print(f"{'Pmp (W)':<12} | {meas_metrics['Pmp']:<18.3f} | {stc_metrics['Pmp']:<18.3f}")
    print(f"{'Vmp (V)':<12} | {meas_metrics['Vmp']:<18.3f} | {stc_metrics['Vmp']:<18.3f}")
    print(f"{'Imp (A)':<12} | {meas_metrics['Imp']:<18.3f} | {stc_metrics['Imp']:<18.3f}")
    print(f"{'Voc (V)':<12} | {meas_metrics['Voc']:<18.3f} | {stc_metrics['Voc']:<18.3f}")
    print(f"{'Isc (A)':<12} | {meas_metrics['Isc']:<18.3f} | {stc_metrics['Isc']:<18.3f}")
    print(f"{'Fill Factor':<12} | {meas_metrics['FF']:<18.3f} | {stc_metrics['FF']:<18.3f}")
    print("=" * 58)
    print(f"\nNormalized curve exported to: {output_csv}\n")

if __name__ == '__main__':
    print("\n--- PV I-V CURVE STC NORMALIZER ---")
    
    # 1. Prompt for Input CSV File
    input_file = input("Enter Input CSV filename [default: 'XXR Module 1 IV.csv']: ").strip()
    if not input_file:
        input_file = "XXR Module 1 IV.csv"
        
    # 2. Prompt for Output CSV File
    output_file = input("Enter Output CSV filename [default: 'iv_stc_normalized.csv']: ").strip()
    if not output_file:
        output_file = "iv_stc_normalized.csv"
        
    # 3. Prompt for Test Date & Central Time
    datetime_prompt = input("Enter test Date & Time in Central Time (e.g. '2024-06-20 14:30') [Leave BLANK if providing manual irradiance]: ").strip()
    
    # 4. Prompt for Cell Temperature
    temp_prompt = input("Enter measured cell temperature in °C [default: 32.0]: ").strip()
    cell_temp = float(temp_prompt) if temp_prompt else 32.0
    
    # 5. Optional Irradiance Override
    irr_prompt = input("Enter measured irradiance in W/m² (Leave BLANK to auto-fetch from NOAA SOLRAD): ").strip()
    manual_g = float(irr_prompt) if irr_prompt else None
    
    print("\nRunning normalization...")
    process_iv_normalization(
        input_csv=input_file,
        output_csv=output_file,
        test_datetime_central=datetime_prompt if datetime_prompt else None,
        G_meas=manual_g,
        T_meas=cell_temp,
        Rs=0.0
    )