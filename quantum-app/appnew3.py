"""
Quantum FFT State Analyzer - Enhanced Version with Test Performance Analysis
A web interface for forward and inverse quantum state analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from itertools import product
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import requests
import tempfile

# Page configuration
st.set_page_config(
    page_title="Quantum FFT State Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# MODEL DOWNLOAD FUNCTION
# ==========================================

@st.cache_resource
def load_model_from_url(url):
    """Download and cache model from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Load model
        model_data = joblib.load(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return model_data
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return None

def check_model_file(filename):
    """Check if a file is a valid model file or just an LFS pointer"""
    if not os.path.exists(filename):
        return False, "File does not exist"
    
    # Check file size - LFS pointer files are typically < 1KB
    file_size = os.path.getsize(filename)
    if file_size < 1000:  # Less than 1KB, likely an LFS pointer
        return False, f"File too small ({file_size} bytes) - likely Git LFS pointer"
    
    # Try to read the file
    try:
        with open(filename, 'rb') as f:
            header = f.read(100)
            # Check if it's an LFS pointer
            if b'version https://git-lfs.github.com' in header:
                return False, "File is a Git LFS pointer, not actual model data"
    except:
        pass
    
    return True, "File appears valid"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .results-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .magnitude-display {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    .phase-display {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .workflow-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 10px 0;
    }
    .workflow-step {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .ml-feature {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2196F3;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
    .error-bad {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .error-good {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# QUANTUM SIMULATION FUNCTIONS (Same as before)
# ==========================================

@st.cache_data
def get_full_basis_6qubits():
    basis_states = list(product([0, 1], repeat=6))
    state_to_idx = {s: i for i, s in enumerate(basis_states)}
    return basis_states, state_to_idx

def commutator(H, rho):
    return -1j * (H @ rho - rho @ H)

def rk4_step(H, rho, dt):
    k1 = dt * commutator(H, rho)
    k2 = dt * commutator(H, rho + 0.5 * k1)
    k3 = dt * commutator(H, rho + 0.5 * k2)
    k4 = dt * commutator(H, rho + k3)
    return rho + (k1 + 2*k2 + 2*k3 + k4) / 6

def initialize_superposition(basis_states, state_to_idx, superposition_dict):
    psi = np.zeros(len(basis_states), complex)
    for state, amp in superposition_dict.items():
        if state in state_to_idx:
            psi[state_to_idx[state]] = amp
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return psi, rho

def calculate_zz_energy(state, k_pattern):
    energy = 0.0
    for (i, j), k_val in k_pattern.items():
        energy += k_val if state[i] == state[j] else -k_val
    return energy

@st.cache_data
def generate_hamiltonian_fullbasis(basis_states, state_to_idx, k_pattern, j_pairs, J_coupling=1.0):
    n = len(basis_states)
    H = np.zeros((n, n), complex)
    
    # Diagonal elements (ZZ interactions)
    for idx, st in enumerate(basis_states):
        H[idx, idx] = calculate_zz_energy(st, k_pattern)

    # Off-diagonal elements (XX+YY interactions)
    for (i, j) in j_pairs:
        for idx, st in enumerate(basis_states):
            if st[i] != st[j]:
                flipped = list(st)
                flipped[i], flipped[j] = st[j], st[i]
                idx2 = state_to_idx[tuple(flipped)]
                H[idx, idx2] += J_coupling
                H[idx2, idx] += J_coupling

    evals, evecs = np.linalg.eigh(H)
    return H, evals, evecs

def analyze_fft_with_peak_fitting(probs, tpoints, qubit_idx, 
                                 min_peak_height=0.1,
                                 peak_type='lorentzian',
                                 fit_window_factor=5,
                                 plot_results=False):
    """Simplified FFT analysis using ONLY CubicSpline - matches training code"""
    
    if np.std(probs[:, qubit_idx]) < 0.001:
        empty_freq = np.linspace(0, 1, 100)
        empty_mag = np.zeros_like(empty_freq)
        return {
            'raw_fft': (empty_freq, empty_mag),
            'peaks': [],
            'fitted_spectrum': empty_mag,
            'message': 'No oscillation detected'
        }
    
    dt = tpoints[1] - tpoints[0]
    sig = probs[:, qubit_idx] - np.mean(probs[:, qubit_idx])
    win = np.hanning(len(sig))
    fft = np.fft.fft(sig * win)
    freq = np.fft.fftfreq(len(fft), dt)
    
    mask = freq >= 0
    pos_freq = freq[mask]
    pos_mag = np.abs(fft[mask])
    
    threshold = min_peak_height * np.max(pos_mag)
    peaks, _ = find_peaks(pos_mag, height=threshold, distance=1)
    
    if len(peaks) > 0 and pos_freq[peaks[0]] < 0.01:
        peaks = peaks[1:]
    
    if len(peaks) == 0:
        return {
            'raw_fft': (pos_freq, pos_mag),
            'peaks': [],
            'fitted_spectrum': np.zeros_like(pos_mag)
        }
    
    spline = CubicSpline(pos_freq, pos_mag)
    fitted_spectrum = spline(pos_freq)
    
    fitted_peaks = []
    
    for peak_idx in peaks:
        window = 20
        left_idx = max(0, peak_idx - window)
        right_idx = min(len(pos_freq), peak_idx + window)
        freq_window = pos_freq[left_idx:right_idx]
        
        if len(freq_window) < 4:
            continue
            
        freq_range = (freq_window[0], freq_window[-1])
        result = minimize_scalar(lambda x: -spline(x), bounds=freq_range, method='bounded')
        
        precise_freq = result.x
        precise_amp = spline(precise_freq)
        
        half_max = precise_amp / 2
        
        left_half = precise_freq
        for f in np.linspace(freq_window[0], precise_freq, 100):
            if spline(f) >= half_max:
                left_half = f
                break
        
        right_half = precise_freq
        for f in np.linspace(precise_freq, freq_window[-1], 100):
            if spline(f) < half_max:
                right_half = f
                break
        
        width = right_half - left_half
        Q_factor = precise_freq / (2 * width) if width > 0 else np.inf
        
        integrated_intensity = precise_amp * width
        
        fitted_peaks.append({
            'frequency': precise_freq,
            'amplitude': precise_amp,
            'width': width,
            'integrated_intensity': integrated_intensity,
            'Q_factor': Q_factor,
            'raw_peak_idx': peak_idx,
            'fit_success': True
        })
    
    fitted_peaks.sort(key=lambda x: x['amplitude'], reverse=True)
    
    while len(fitted_peaks) < 5:
        fitted_peaks.append({
            'frequency': 0.0,
            'amplitude': 0.0,
            'width': 0.0,
            'integrated_intensity': 0.0,
            'Q_factor': 0.0,
            'raw_peak_idx': -1,
            'fit_success': False
        })
    
    if len(fitted_peaks) > 5:
        fitted_peaks = fitted_peaks[:5]
    
    return {
        'raw_fft': (pos_freq, pos_mag),
        'peaks': fitted_peaks,
        'fitted_spectrum': fitted_spectrum
    }

def run_quantum_simulation(a_complex, b_complex, c_complex, k01, k23, k45, j_coupling):
    """Run the quantum simulation and return FFT peaks"""
    
    dt = 0.01
    t_max = 50
    
    k_pattern = {
        (0,1): k01,
        (2,3): k23,
        (4,5): k45
    }
    j_pairs = [(0,2), (2,4)]
    
    basis_states, state_to_idx = get_full_basis_6qubits()
    projectors = [np.diag([s[q] for s in basis_states]) for q in range(6)]
    
    state_A = (1,1,0,0,0,0)
    state_B = (1,0,0,1,0,0)
    state_C = (1,0,0,0,0,1)
    
    superposition_dict = {
        state_A: a_complex,
        state_B: b_complex,
        state_C: c_complex
    }
    
    psi0, rho0 = initialize_superposition(basis_states, state_to_idx, superposition_dict)
    
    H, evals, evecs = generate_hamiltonian_fullbasis(
        basis_states, state_to_idx, k_pattern, j_pairs, j_coupling
    )
    
    times = np.arange(0, t_max + dt, dt)
    probs = np.zeros((len(times), 6))
    
    for q in range(6):
        probs[0, q] = np.real(np.trace(rho0 @ projectors[q]))
    
    rho = rho0.copy()
    for i in range(1, len(times)):
        rho = rk4_step(H, rho, dt)
        for q in range(6):
            probs[i, q] = np.real(np.trace(rho @ projectors[q]))
    
    peak_data = analyze_fft_with_peak_fitting(probs, times, 4, min_peak_height=0.05)
    
    peaks_list = peak_data['peaks']
    peaks_sorted = sorted(peaks_list, key=lambda x: x['amplitude'], reverse=True)
    
    peaks = []
    for peak in peaks_sorted[:5]:
        peaks.append({
            'freq': float(peak['frequency']),
            'amp': float(peak['amplitude'])
        })
    
    while len(peaks) < 5:
        peaks.append({'freq': 0.0, 'amp': 0.0})
    
    freq, mag = peak_data['raw_fft']
    
    return peaks[:5], probs, times, (freq, mag)

def draw_network_diagram():
    """Draw the 6-qubit chain topology in a 2x3 grid layout"""
    fig = go.Figure()
    
    qubit_x = [0, 2, 4, 0, 2, 4]
    qubit_y = [1, 1, 1, 0, 0, 0]
    qubit_labels = ['0', '2', '4', '1', '3', '5']
    
    qubit_colors = ['lightblue', 'lightblue', 'lightblue', 'lightpink', 'lightpink', 'lightpink']
    
    fig.add_trace(go.Scatter(
        x=qubit_x, y=qubit_y,
        mode='markers+text',
        marker=dict(size=60, color=qubit_colors, line=dict(width=2, color='darkgray')),
        text=qubit_labels,
        textposition="middle center",
        textfont=dict(size=16, color='black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    j_connections = [(0, 1), (1, 2)]
    for i, (start, end) in enumerate(j_connections):
        fig.add_trace(go.Scatter(
            x=[qubit_x[start], qubit_x[end]], 
            y=[qubit_y[start], qubit_y[end]],
            mode='lines',
            line=dict(width=2, color='darkgreen'),
            showlegend=False,
            hoverinfo='skip'
        ))
        mid_x = (qubit_x[start] + qubit_x[end]) / 2
        fig.add_annotation(
            x=mid_x, y=1.2, 
            text=f"J<sub>{qubit_labels[start]},{qubit_labels[end]}</sub>", 
            showarrow=False, 
            font=dict(size=12, color='darkgreen')
        )
    
    k_pairs = [(0, 3), (1, 4), (2, 5)]
    k_labels = ['K<sub>0,1</sub>', 'K<sub>2,3</sub>', 'K<sub>4,5</sub>']
    for i, (top, bottom) in enumerate(k_pairs):
        fig.add_trace(go.Scatter(
            x=[qubit_x[top], qubit_x[bottom]], 
            y=[qubit_y[top], qubit_y[bottom]],
            mode='lines',
            line=dict(width=4, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_annotation(
            x=qubit_x[top] - 0.3, 
            y=(qubit_y[top] + qubit_y[bottom]) / 2,
            text=k_labels[i], 
            showarrow=False, 
            font=dict(size=12, color='red')
        )
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(width=4, color='red'),
        name='k-coupling (ZZ)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(width=2, color='darkgreen'),
        name='J-coupling (XX+YY)'
    ))
    
    fig.update_layout(
        title=dict(
            text="6-Qubit Network Topology",
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-1, 5]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-0.5, 1.5]
        ),
        height=300,
        margin=dict(l=50, r=150, t=50, b=50),
        plot_bgcolor='white'
    )
    
    return fig

# ==========================================
# TEST PERFORMANCE ANALYSIS FUNCTIONS
# ==========================================

@st.cache_data
def load_test_dataset(file_path=None):
    """Load test dataset from file or default location"""
    if file_path is None:
        # Try default locations
        default_files = [
            "quantum_fft_test_20250607_195702.csv",
            "ml_data_cubic_spline/quantum_fft_test_20250616_214918.csv",
            "quantum_fft_test.csv"
        ]
        
        for file in default_files:
            if os.path.exists(file):
                file_path = file
                break
        
        if file_path is None:
            return None, "No test dataset found. Please upload one."
    
    try:
        df = pd.read_csv(file_path)
        return df, f"Loaded {len(df)} samples from {file_path}"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def prepare_features_for_prediction(df):
    """Prepare features exactly as in training"""
    # Sort peaks by amplitude for each sample
    sorted_data = []
    
    for _, row in df.iterrows():
        # Get peak data
        peaks = []
        for i in range(1, 6):
            freq = row[f'peak{i}_freq']
            amp = row[f'peak{i}_amp']
            peaks.append((freq, amp))
        
        # Sort by amplitude (descending)
        peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
        
        # Extract sorted frequencies and amplitudes
        sorted_freqs = [p[0] for p in peaks_sorted]
        sorted_amps = [p[1] for p in peaks_sorted]
        
        # Basic features (10)
        features = []
        for i in range(5):
            features.extend([sorted_freqs[i], sorted_amps[i]])
        
        # Engineered features (5)
        total_power = sum(sorted_amps)
        n_peaks = sum(1 for f in sorted_freqs if f > 0)
        max_freq = max(sorted_freqs)
        max_amp = max(sorted_amps)
        freq_spread = max(sorted_freqs) - min(sorted_freqs)
        
        features.extend([total_power, n_peaks, max_freq, max_amp, freq_spread])
        
        sorted_data.append(features)
    
    return np.array(sorted_data)

def calculate_true_magnitudes(df):
    """Calculate true magnitudes from complex amplitudes"""
    a_mag = np.sqrt(df['a_real']**2 + df['a_imag']**2)
    b_mag = np.sqrt(df['b_real']**2 + df['b_imag']**2)
    c_mag = np.sqrt(df['c_real']**2 + df['c_imag']**2)
    
    return np.column_stack([a_mag, b_mag, c_mag])

@st.cache_data
def run_bulk_predictions(test_df, model_path):
    """Run predictions on entire test dataset"""
    try:
        # Load model
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Prepare features
        X_test = prepare_features_for_prediction(test_df)
        X_test_scaled = scaler.transform(X_test)
        
        # Get true magnitudes
        y_true = calculate_true_magnitudes(test_df)
        
        # Make predictions
        if isinstance(model, dict):
            # Ensemble model
            predictions = {}
            for name, m in model.items():
                if hasattr(m, 'predict'):
                    predictions[name] = m.predict(X_test_scaled)
            
            # Weighted ensemble
            approach = model_data.get('approach', 'Ensemble (Weighted)')
            if 'Weighted' in approach:
                weights = {'rf': 0.4, 'et': 0.3, 'nn': 0.3}
                y_pred = np.zeros_like(list(predictions.values())[0])
                total_weight = 0
                for name, pred in predictions.items():
                    weight = weights.get(name, 1.0/len(predictions))
                    y_pred += pred * weight
                    total_weight += weight
                y_pred /= total_weight
            else:
                # Simple average
                y_pred = np.mean(list(predictions.values()), axis=0)
        else:
            # Single model
            y_pred = model.predict(X_test_scaled)
        
        # Apply calibration if available
        if model_data.get('calibration_factors'):
            cal_factors = model_data['calibration_factors']
            for i in range(3):
                y_pred[:, i] *= cal_factors[i]
        
        return y_true, y_pred, model_data
        
    except Exception as e:
        st.error(f"Error in bulk prediction: {str(e)}")
        return None, None, None

def analyze_errors(y_true, y_pred):
    """Comprehensive error analysis"""
    # Calculate various error metrics
    mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)
    rmse_per_sample = np.sqrt(np.mean((y_true - y_pred)**2, axis=1))
    
    # Relative errors (avoid division by zero)
    rel_errors = np.abs(y_true - y_pred) / (y_true + 1e-10) * 100
    rel_error_per_sample = np.mean(rel_errors, axis=1)
    
    # Per-magnitude errors
    mae_per_mag = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_mag = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    
    # R² for each magnitude
    r2_per_mag = []
    for i in range(3):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i])**2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i]))**2)
        r2_per_mag.append(1 - (ss_res / ss_tot))
    
    # Average magnitudes per sample
    avg_true_mag = np.mean(y_true, axis=1)
    
    return {
        'mae_per_sample': mae_per_sample,
        'rmse_per_sample': rmse_per_sample,
        'rel_error_per_sample': rel_error_per_sample,
        'mae_per_mag': mae_per_mag,
        'rmse_per_mag': rmse_per_mag,
        'r2_per_mag': r2_per_mag,
        'avg_true_mag': avg_true_mag,
        'rel_errors': rel_errors
    }

def show_test_performance_analysis():
    """Display the test performance analysis tab"""
    
    st.header("📈 Test Performance Analysis")
    st.markdown("### Comprehensive ML Model Evaluation on Test Dataset")
    
    # Dataset loading section
    st.markdown("---")
    st.subheader("📁 Dataset Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload test dataset (CSV)", 
            type=['csv'],
            help="Upload your test dataset or use default if available"
        )
        
        # Load dataset
        if uploaded_file is not None:
            test_df = pd.read_csv(uploaded_file)
            dataset_msg = f"Uploaded dataset: {len(test_df)} samples"
        else:
            test_df, dataset_msg = load_test_dataset()
        
        if test_df is not None:
            st.success(dataset_msg)
            
            # Show dataset info
            st.markdown("**Dataset Info:**")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{len(test_df)}</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
            with col_info2:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{len(test_df.columns)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
            with col_info3:
                # Count samples with valid peaks
                valid_samples = sum(1 for _, row in test_df.iterrows() 
                                  if any(row[f'peak{i}_freq'] > 0 for i in range(1, 6)))
                st.markdown(f'<div class="metric-box"><div class="metric-value">{valid_samples}</div><div class="metric-label">Valid Peaks</div></div>', unsafe_allow_html=True)
        else:
            st.error(dataset_msg)
            st.info("Please upload a test dataset to continue with the analysis.")
            return
    
    with col2:
        # Dataset preview
        if test_df is not None:
            st.markdown("**Dataset Preview:**")
            st.dataframe(test_df.head(3), use_container_width=True)
    
    # Model selection and bulk prediction
    st.markdown("---")
    st.subheader("🤖 Bulk Prediction Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Model path selection
        model_paths = []
        
        # Look for available models
        possible_paths = [
            'quantum_simple_nn_20250623_215653.pkl',
            'quantum-app/quantum_simple_nn_20250623_215653.pkl',
            'quantum_inverse_model.pkl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_paths.append(path)
        
        if model_paths:
            selected_model = st.selectbox("Select model for analysis:", model_paths)
        else:
            st.error("No valid model files found!")
            return
    
    with col2:
        # Analysis controls
        run_analysis = st.button("🚀 Run Full Analysis", use_container_width=True)
        
        if run_analysis:
            st.session_state.run_bulk_analysis = True
    
    # Run bulk analysis if requested
    if run_analysis or st.session_state.get('run_bulk_analysis', False):
        with st.spinner("Running bulk predictions on test dataset..."):
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading model and preparing features...")
            progress_bar.progress(20)
            
            # Run predictions
            y_true, y_pred, model_data = run_bulk_predictions(test_df, selected_model)
            progress_bar.progress(60)
            
            if y_true is not None and y_pred is not None:
                status_text.text("Analyzing prediction errors...")
                progress_bar.progress(80)
                
                # Calculate comprehensive error analysis
                error_analysis = analyze_errors(y_true, y_pred)
                progress_bar.progress(100)
                
                status_text.text("Analysis complete!")
                
                # Store results in session state
                st.session_state.bulk_results = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'error_analysis': error_analysis,
                    'model_data': model_data,
                    'test_df': test_df
                }
                
                st.success("✅ Bulk analysis completed successfully!")
            else:
                st.error("❌ Failed to run bulk analysis")
                return
    
    # Display results if available
    if 'bulk_results' in st.session_state:
        results = st.session_state.bulk_results
        y_true = results['y_true']
        y_pred = results['y_pred']
        error_analysis = results['error_analysis']
        model_data = results['model_data']
        
        # Performance metrics summary
        st.markdown("---")
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            overall_mae = np.mean(error_analysis['mae_per_sample'])
            st.markdown(f'<div class="metric-box"><div class="metric-value">{overall_mae:.4f}</div><div class="metric-label">Overall MAE</div></div>', unsafe_allow_html=True)
        
        with col2:
            overall_rmse = np.mean(error_analysis['rmse_per_sample'])
            st.markdown(f'<div class="metric-box"><div class="metric-value">{overall_rmse:.4f}</div><div class="metric-label">Overall RMSE</div></div>', unsafe_allow_html=True)
        
        with col3:
            overall_rel_error = np.mean(error_analysis['rel_error_per_sample'])
            st.markdown(f'<div class="metric-box"><div class="metric-value">{overall_rel_error:.1f}%</div><div class="metric-label">Avg Rel Error</div></div>', unsafe_allow_html=True)
        
        with col4:
            overall_r2 = np.mean(error_analysis['r2_per_mag'])
            st.markdown(f'<div class="metric-box"><div class="metric-value">{overall_r2:.4f}</div><div class="metric-label">Avg R²</div></div>', unsafe_allow_html=True)
        
        with col5:
            good_predictions = np.sum(error_analysis['rel_error_per_sample'] < 10)
            good_percent = good_predictions / len(y_true) * 100
            st.markdown(f'<div class="metric-box"><div class="metric-value">{good_percent:.1f}%</div><div class="metric-label">< 10% Error</div></div>', unsafe_allow_html=True)
        
        # Per-magnitude breakdown
        st.markdown("**Per-Magnitude Performance:**")
        
        mag_col1, mag_col2, mag_col3 = st.columns(3)
        magnitude_names = ['|a|', '|b|', '|c|']
        
        for i, (col, name) in enumerate(zip([mag_col1, mag_col2, mag_col3], magnitude_names)):
            with col:
                mae_val = error_analysis['mae_per_mag'][i]
                r2_val = error_analysis['r2_per_mag'][i]
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-weight: bold; margin-bottom: 10px;">{name}</div>
                    <div>MAE: {mae_val:.4f}</div>
                    <div>R²: {r2_val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Error distribution visualizations
        st.markdown("---")
        st.subheader("📈 Error Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # MAE histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(error_analysis['mae_per_sample'], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(error_analysis['mae_per_sample']), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(error_analysis["mae_per_sample"]):.4f}')
            ax.axvline(np.median(error_analysis['mae_per_sample']), color='green', linestyle='--', 
                      label=f'Median: {np.median(error_analysis["mae_per_sample"]):.4f}')
            ax.set_xlabel('Mean Absolute Error per Sample')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of MAE Across Test Samples')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Relative error histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(error_analysis['rel_error_per_sample'], bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(np.mean(error_analysis['rel_error_per_sample']), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(error_analysis["rel_error_per_sample"]):.1f}%')
            ax.axvline(np.median(error_analysis['rel_error_per_sample']), color='green', linestyle='--', 
                      label=f'Median: {np.median(error_analysis["rel_error_per_sample"]):.1f}%')
            ax.set_xlabel('Relative Error per Sample (%)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Relative Error Across Test Samples')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, min(100, np.percentile(error_analysis['rel_error_per_sample'], 95)))
            st.pyplot(fig)
        
        # Scatter plots - Predicted vs True
        st.markdown("### 🎯 Predicted vs True Magnitude Scatter Plots")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (ax, name) in enumerate(zip(axes, magnitude_names)):
            # Scatter plot
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = 0
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Add error bands
            ax.fill_between([min_val, max_val], 
                           [min_val*0.9, max_val*0.9], 
                           [min_val*1.1, max_val*1.1], 
                           alpha=0.2, color='gray', label='±10% Error')
            
            # Metrics
            mae = error_analysis['mae_per_mag'][i]
            r2 = error_analysis['r2_per_mag'][i]
            
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name}\nMAE={mae:.4f}, R²={r2:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, max_val * 1.05)
            ax.set_ylim(0, max_val * 1.05)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Performance vs magnitude analysis
        st.markdown("### 📉 Performance vs Magnitude Analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Error vs average magnitude
        avg_mags = error_analysis['avg_true_mag']
        mae_samples = error_analysis['mae_per_sample']
        rel_error_samples = error_analysis['rel_error_per_sample']
        
        # Bin by magnitude ranges
        mag_bins = np.linspace(0, avg_mags.max(), 10)
        bin_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
        
        binned_mae = []
        binned_rel_error = []
        
        for i in range(len(mag_bins)-1):
            mask = (avg_mags >= mag_bins[i]) & (avg_mags < mag_bins[i+1])
            if np.sum(mask) > 0:
                binned_mae.append(np.mean(mae_samples[mask]))
                binned_rel_error.append(np.mean(rel_error_samples[mask]))
            else:
                binned_mae.append(np.nan)
                binned_rel_error.append(np.nan)
        
        # Plot MAE vs magnitude
        ax1.plot(bin_centers, binned_mae, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Average True Magnitude |a|, |b|, |c|')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('MAE vs State Magnitude')
        ax1.grid(True, alpha=0.3)
        
        # Add regions
        ax1.axvspan(0, 0.5, alpha=0.2, color='red', label='High Error Region')
        ax1.axvspan(0.5, 1.0, alpha=0.2, color='yellow', label='Moderate Error')
        ax1.axvspan(1.0, avg_mags.max(), alpha=0.2, color='green', label='Low Error Region')
        ax1.legend()
        
        # Plot relative error vs magnitude
        ax2.plot(bin_centers, binned_rel_error, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Average True Magnitude |a|, |b|, |c|')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Relative Error vs State Magnitude')
        ax2.grid(True, alpha=0.3)
        
        # Add regions
        ax2.axvspan(0, 0.5, alpha=0.2, color='red')
        ax2.axvspan(0.5, 1.0, alpha=0.2, color='yellow')
        ax2.axvspan(1.0, avg_mags.max(), alpha=0.2, color='green')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Outlier analysis
        st.markdown("---")
        st.subheader("🔍 Outlier Analysis")
        
        # Find worst and best cases
        worst_indices = np.argsort(error_analysis['rel_error_per_sample'])[-10:][::-1]
        best_indices = np.argsort(error_analysis['rel_error_per_sample'])[:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Worst 10 Predictions:**")
            worst_data = []
            for idx in worst_indices:
                worst_data.append({
                    'Sample': int(idx),
                    'Rel Error (%)': f"{error_analysis['rel_error_per_sample'][idx]:.1f}",
                    'MAE': f"{error_analysis['mae_per_sample'][idx]:.4f}",
                    'Avg Magnitude': f"{error_analysis['avg_true_mag'][idx]:.3f}"
                })
            
            worst_df = pd.DataFrame(worst_data)
            st.dataframe(worst_df, use_container_width=True)
        
        with col2:
            st.markdown("**🟢 Best 10 Predictions:**")
            best_data = []
            for idx in best_indices:
                best_data.append({
                    'Sample': int(idx),
                    'Rel Error (%)': f"{error_analysis['rel_error_per_sample'][idx]:.1f}",
                    'MAE': f"{error_analysis['mae_per_sample'][idx]:.4f}",
                    'Avg Magnitude': f"{error_analysis['avg_true_mag'][idx]:.3f}"
                })
            
            best_df = pd.DataFrame(best_data)
            st.dataframe(best_df, use_container_width=True)
        
        # Statistical summary
        st.markdown("### 📋 Statistical Summary")
        
        percentiles = [50, 75, 90, 95, 99]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**MAE Percentiles:**")
            mae_stats = []
            for p in percentiles:
                mae_stats.append({
                    'Percentile': f"{p}th",
                    'MAE': f"{np.percentile(error_analysis['mae_per_sample'], p):.4f}"
                })
            st.dataframe(pd.DataFrame(mae_stats), use_container_width=True)
        
        with col2:
            st.markdown("**Relative Error Percentiles:**")
            rel_stats = []
            for p in percentiles:
                rel_stats.append({
                    'Percentile': f"{p}th",
                    'Rel Error (%)': f"{np.percentile(error_analysis['rel_error_per_sample'], p):.1f}"
                })
            st.dataframe(pd.DataFrame(rel_stats), use_container_width=True)
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        insights = []
        
        # Performance by magnitude ranges
        low_mag_mask = error_analysis['avg_true_mag'] < 0.5
        high_mag_mask = error_analysis['avg_true_mag'] > 1.5
        
        if np.sum(low_mag_mask) > 0:
            low_mag_error = np.mean(error_analysis['rel_error_per_sample'][low_mag_mask])
            insights.append(f"🔴 Low magnitude states (< 0.5): {low_mag_error:.1f}% average relative error")
        
        if np.sum(high_mag_mask) > 0:
            high_mag_error = np.mean(error_analysis['rel_error_per_sample'][high_mag_mask])
            insights.append(f"🟢 High magnitude states (> 1.5): {high_mag_error:.1f}% average relative error")
        
        # Error distribution
        excellent_count = np.sum(error_analysis['rel_error_per_sample'] < 5)
        good_count = np.sum((error_analysis['rel_error_per_sample'] >= 5) & (error_analysis['rel_error_per_sample'] < 10))
        poor_count = np.sum(error_analysis['rel_error_per_sample'] >= 20)
        
        insights.extend([
            f"🎯 {excellent_count} samples ({excellent_count/len(y_true)*100:.1f}%) have < 5% relative error",
            f"✅ {good_count} samples ({good_count/len(y_true)*100:.1f}%) have 5-10% relative error",
            f"⚠️ {poor_count} samples ({poor_count/len(y_true)*100:.1f}%) have > 20% relative error"
        ])
        
        # Best performing magnitude
        best_mag_idx = np.argmin(error_analysis['mae_per_mag'])
        worst_mag_idx = np.argmax(error_analysis['mae_per_mag'])
        
        insights.extend([
            f"🏆 Best predicted magnitude: {magnitude_names[best_mag_idx]} (MAE: {error_analysis['mae_per_mag'][best_mag_idx]:.4f})",
            f"📉 Most challenging magnitude: {magnitude_names[worst_mag_idx]} (MAE: {error_analysis['mae_per_mag'][worst_mag_idx]:.4f})"
        ])
        
        for insight in insights:
            if "🔴" in insight or "⚠️" in insight or "📉" in insight:
                st.markdown(f'<div class="error-bad">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-good">{insight}</div>', unsafe_allow_html=True)

# ==========================================
# WORKFLOW EXPLANATION FUNCTIONS (Same as before)
# ==========================================

def show_workflow_explanation():
    """Display the workflow explanation tab"""
    
    st.header("🔄 How the Quantum FFT State Analyzer Works")
    
    # Overview
    st.markdown("### 📋 Overview")
    st.info("""
    This application demonstrates a quantum inverse problem: recovering quantum state information from 
    frequency measurements. It combines quantum simulation with machine learning to solve both the 
    forward problem (state → frequencies) and the inverse problem (frequencies → state).
    """)
    
    # Create columns for side-by-side workflow
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ➡️ Forward Problem Workflow")
        st.markdown('<div class="workflow-box">', unsafe_allow_html=True)
        
        # Step 1
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 1: Define Quantum State**")
        st.markdown("Input: Complex amplitudes a, b, c")
        st.latex(r"|ψ⟩ = a|110000⟩ + b|100100⟩ + c|100001⟩")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 2
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 2: Build Hamiltonian**")
        st.markdown("- ZZ interactions (k-couplings)")
        st.markdown("- XX+YY interactions (J-couplings)")
        st.latex(r"H = \sum_{i,j} k_{ij}Z_iZ_j + J\sum_{⟨i,j⟩}(X_iX_j + Y_iY_j)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 3
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 3: Time Evolution**")
        st.markdown("- Solve Schrödinger equation")
        st.markdown("- Use Runge-Kutta 4th order")
        st.latex(r"i\hbar\frac{d\rho}{dt} = [H, \rho]")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 4
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 4: Extract Dynamics**")
        st.markdown("- Monitor qubit 4 probability P₄(t)")
        st.markdown("- Sample over 50 time units")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 5
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 5: FFT Analysis**")
        st.markdown("- Apply windowed FFT")
        st.markdown("- Find peaks using cubic spline")
        st.markdown("- Sort by amplitude (largest first)")
        st.markdown("Output: Top 5 frequency peaks")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⬅️ Inverse Problem Workflow")
        st.markdown('<div class="workflow-box">', unsafe_allow_html=True)
        
        # Step 1
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 1: Input FFT Peaks**")
        st.markdown("Input: 5 frequency peaks (sorted by amplitude)")
        st.markdown("- Peak frequencies (Hz)")
        st.markdown("- Peak amplitudes")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 2
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 2: Feature Engineering**")
        st.markdown("Basic features (10):")
        st.markdown("- 5 frequencies + 5 amplitudes")
        st.markdown("Engineered features (5):")
        st.markdown("- Total power, # peaks, max freq/amp, freq spread")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 3
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 3: Feature Scaling**")
        st.markdown("- StandardScaler normalization")
        st.markdown("- Same scaling as training data")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 4
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 4: ML Prediction**")
        st.markdown("Model options:")
        st.markdown("- Neural Network (3 layers)")
        st.markdown("- Random Forest")
        st.markdown("- Ensemble (weighted average)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 5
        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
        st.markdown("**Step 5: Output Magnitudes**")
        st.markdown("Output: |a|, |b|, |c|")
        st.markdown("- Magnitudes only (phase lost)")
        st.markdown("- Confidence score")
        st.markdown("- Error metrics if truth known")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ML Model Details
    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Model Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Training Data")
        st.markdown('<div class="ml-feature">', unsafe_allow_html=True)
        st.markdown("**Dataset Generation:**")
        st.markdown("- 5000 random quantum states")
        st.markdown("- Complex amplitudes ∈ [-3, 3]")
        st.markdown("- 80/20 train/test split")
        st.markdown("- Full quantum simulation for each")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Model Architecture")
        st.markdown('<div class="ml-feature">', unsafe_allow_html=True)
        st.markdown("**Neural Network:**")
        st.markdown("- Input: 15 features")
        st.markdown("- Hidden: [150, 100, 50] neurons")
        st.markdown("- Output: 3 magnitudes")
        st.markdown("- Activation: ReLU")
        st.markdown("- Optimizer: Adam")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### Performance")
        st.markdown('<div class="ml-feature">', unsafe_allow_html=True)
        st.markdown("**Typical Results:**")
        st.markdown("- Test R² > 0.996")
        st.markdown("- MAE < 0.025")
        st.markdown("- Relative error < 3%")
        st.markdown("- Confidence based on # peaks")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ML Training Results
    st.markdown("---")
    st.markdown("### 📊 ML Training Results")
    
    # Create a nice table for the results
    results_data = {
        'Approach': ['Neural Network', 'NN Calibrated', 'Ensemble (Average)', 'Ensemble (Weighted)'],
        'MAE': [0.0272, 0.0271, 0.0241, 0.0237],
        'RMSE': [0.0550, 0.0549, 0.0425, 0.0418],
        'R²': [0.9960, 0.9960, 0.9976, 0.9977],
        'Rel Error': ['3.21%', '3.20%', '3.04%', '3.00%']
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Style the dataframe
    st.markdown("**Final Model Comparison:**")
    st.dataframe(
        results_df.style.highlight_min(subset=['MAE', 'RMSE', 'Rel Error'], color='lightgreen')
                       .highlight_max(subset=['R²'], color='lightgreen')
                       .format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R²': '{:.4f}'}),
        use_container_width=True
    )
    
    st.success("✅ **Best Approach: Ensemble (Weighted)** - Achieves lowest error and highest R²")

# ==========================================
# MAIN APP SECTION (Same as before but condensed)
# ==========================================

def show_main_app():
    """Display the main application interface"""
    
    # Network diagram
    with st.container():
        st.plotly_chart(draw_network_diagram(), use_container_width=True)

    st.markdown("---")
    st.markdown("### Quick Forward/Inverse Demo")
    st.info("For detailed analysis, use the **Test Performance Analysis** tab above.")

    # Simplified demo version - show core functionality
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Forward Demo:** Quantum State → FFT Peaks")
        if st.button("🔄 Run Demo Simulation"):
            # Quick demo with fixed values
            a_complex = 0.148 + 2.026j
            b_complex = -2.004 + 0.294j  
            c_complex = 1.573 - 2.555j
            
            with st.spinner("Running simulation..."):
                peaks, _, _, _ = run_quantum_simulation(a_complex, b_complex, c_complex, 0.7, 1.0, 1.9, 0.5)
                
            st.markdown("**Detected Peaks:**")
            for i, peak in enumerate(peaks):
                if peak['freq'] > 0:
                    st.write(f"• Peak {i+1}: {peak['freq']:.3f} Hz (amp: {peak['amp']:.3f})")

    with col2:
        st.markdown("**Inverse Demo:** FFT Peaks → State Magnitudes")
        st.info("Use the full interface in previous versions or the Test Performance Analysis tab for complete inverse functionality.")

# ==========================================
# STREAMLIT APP MAIN STRUCTURE
# ==========================================

# Initialize session state
if 'k_values_locked' not in st.session_state:
    st.session_state.k_values_locked = True
if 'forward_results' not in st.session_state:
    st.session_state.forward_results = None
if 'inverse_results' not in st.session_state:
    st.session_state.inverse_results = None
if 'forward_magnitudes' not in st.session_state:
    st.session_state.forward_magnitudes = None
if 'run_bulk_analysis' not in st.session_state:
    st.session_state.run_bulk_analysis = False

# Title and description
st.title("🔬 Quantum FFT State Analyzer")

# Create tabs - UPDATED with new tab
tab1, tab2, tab3 = st.tabs(["🎯 Main Application", "📈 Test Performance Analysis", "🔄 How It Works"])

with tab1:
    show_main_app()
    
    # Status bar
    st.markdown("---")
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        # Check local model file
        model_file = 'quantum_simple_nn_20250623_215653.pkl'
        if os.path.exists(model_file):
            is_valid, msg = check_model_file(model_file)
            if is_valid:
                st.success("Status: Ready | Model file found and valid ✓")
            else:
                st.warning(f"Status: Model issue | {msg}")
        else:
            st.warning("Status: Limited | Model file not found - see sidebar for solutions")
    with status_col2:
        st.info(f"k-values {'locked 🔒' if st.session_state.k_values_locked else 'unlocked 🔓'}")

with tab2:
    show_test_performance_analysis()  # NEW TAB

with tab3:
    show_workflow_explanation()

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions & Features")
    
    with st.expander("⚡ Quick Start", expanded=True):
        st.markdown("""
        ### 🎯 Main App
        - Simple forward/inverse demo
        - Real-time quantum simulation
        
        ### 📈 Test Analysis  
        - **Upload test dataset** or use default
        - **Run bulk predictions** on all samples
        - **Comprehensive error analysis**
        - **Visual performance insights**
        
        ### 🔄 How It Works
        - Detailed workflow explanation
        - Technical implementation
        """)
    
    with st.expander("📊 Test Analysis Features"):
        st.markdown("""
        **Performance Metrics:**
        - MAE, RMSE, R² per magnitude
        - Error distribution histograms
        - Relative error analysis
        
        **Advanced Analysis:**
        - Performance vs magnitude ranges
        - Best/worst case identification
        - Statistical percentiles
        - Outlier detection
        
        **Visualizations:**
        - Predicted vs true scatter plots
        - Error distribution curves
        - Performance degradation analysis
        """)
    
    with st.expander("⚠️ Important Notes"):
        st.markdown("""
        - **Peak ordering is CRITICAL** - always by amplitude!
        - Large datasets may take time to process
        - Results are cached for performance
        - Low magnitude states show higher errors
        """)
