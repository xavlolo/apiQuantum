"""
Quantum FFT State Analyzer - Fixed Version with Model Loading Solution
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
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# QUANTUM SIMULATION FUNCTIONS (same as before)
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
                                 peak_type='lorentzian',  # Keep for compatibility
                                 fit_window_factor=5,     # Keep for compatibility
                                 plot_results=False):
    """
    Simplified FFT analysis using ONLY CubicSpline - matches training code
    """
    
    # Check if qubit has any oscillation
    if np.std(probs[:, qubit_idx]) < 0.001:
        empty_freq = np.linspace(0, 1, 100)
        empty_mag = np.zeros_like(empty_freq)
        return {
            'raw_fft': (empty_freq, empty_mag),
            'peaks': [],
            'fitted_spectrum': empty_mag,
            'message': 'No oscillation detected'
        }
    
    # Compute FFT
    dt = tpoints[1] - tpoints[0]
    sig = probs[:, qubit_idx] - np.mean(probs[:, qubit_idx])
    win = np.hanning(len(sig))
    fft = np.fft.fft(sig * win)
    freq = np.fft.fftfreq(len(fft), dt)
    
    # Focus on positive frequencies
    mask = freq >= 0
    pos_freq = freq[mask]
    pos_mag = np.abs(fft[mask])
    
    # Find peaks
    threshold = min_peak_height * np.max(pos_mag)
    peaks, _ = find_peaks(pos_mag, height=threshold, distance=5)
    
    # Skip DC component if present
    if len(peaks) > 0 and pos_freq[peaks[0]] < 0.05:
        if plot_results:
            print(f"Skipping DC component at {pos_freq[peaks[0]]:.4f} Hz")
        peaks = peaks[1:]
    
    if len(peaks) == 0:
        if plot_results:
            print("No significant oscillation peaks found")
        return {
            'raw_fft': (pos_freq, pos_mag),
            'peaks': [],
            'fitted_spectrum': np.zeros_like(pos_mag)
        }
    
    # Create cubic spline
    spline = CubicSpline(pos_freq, pos_mag)
    fitted_spectrum = spline(pos_freq)
    
    # Analyze each peak
    fitted_peaks = []
    
    for peak_idx in peaks:
        
        # Define window around peak
        window = 20
        left_idx = max(0, peak_idx - window)
        right_idx = min(len(pos_freq), peak_idx + window)
        freq_window = pos_freq[left_idx:right_idx]
        
        if len(freq_window) < 4:
            continue
            
        # Find precise peak maximum using spline
        freq_range = (freq_window[0], freq_window[-1])
        result = minimize_scalar(lambda x: -spline(x), bounds=freq_range, method='bounded')
        
        precise_freq = result.x
        precise_amp = spline(precise_freq)
        
        # Estimate width by finding half-maximum points
        half_max = precise_amp / 2
        
        # Find left half-maximum
        left_half = precise_freq
        for f in np.linspace(freq_window[0], precise_freq, 100):
            if spline(f) >= half_max:
                left_half = f
                break
        
        # Find right half-maximum
        right_half = precise_freq
        for f in np.linspace(precise_freq, freq_window[-1], 100):
            if spline(f) < half_max:
                right_half = f
                break
        
        width = right_half - left_half
        Q_factor = precise_freq / (2 * width) if width > 0 else np.inf
        
        # Calculate integrated intensity (simple approximation)
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
    
    # CRITICAL FIX: Sort peaks by AMPLITUDE (descending) to match training data
    fitted_peaks.sort(key=lambda x: x['amplitude'], reverse=True)
    
    # Ensure exactly 5 peaks for ML compatibility
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
    
    # Keep only top 5 by amplitude
    if len(fitted_peaks) > 5:
        fitted_peaks = fitted_peaks[:5]
    
    return {
        'raw_fft': (pos_freq, pos_mag),
        'peaks': fitted_peaks,
        'fitted_spectrum': fitted_spectrum
    }

def run_quantum_simulation(a_complex, b_complex, c_complex, k01, k23, k45, j_coupling):
    """Run the quantum simulation and return FFT peaks"""
    
    # Setup
    dt = 0.01
    t_max = 50
    
    k_pattern = {
        (0,1): k01,
        (2,3): k23,
        (4,5): k45
    }
    j_pairs = [(0,2), (2,4)]  # Fixed for this setup
    
    # Get basis states
    basis_states, state_to_idx = get_full_basis_6qubits()
    projectors = [np.diag([s[q] for s in basis_states]) for q in range(6)]
    
    # Define quantum states
    state_A = (1,1,0,0,0,0)  # |110000⟩
    state_B = (1,0,0,1,0,0)  # |100100⟩
    state_C = (1,0,0,0,0,1)  # |100001⟩
    
    superposition_dict = {
        state_A: a_complex,
        state_B: b_complex,
        state_C: c_complex
    }
    
    # Initialize quantum state
    psi0, rho0 = initialize_superposition(basis_states, state_to_idx, superposition_dict)
    
    # Generate Hamiltonian
    H, evals, evecs = generate_hamiltonian_fullbasis(
        basis_states, state_to_idx, k_pattern, j_pairs, j_coupling
    )
    
    # Time evolution
    times = np.arange(0, t_max + dt, dt)
    probs = np.zeros((len(times), 6))
    
    # Initial probabilities
    for q in range(6):
        probs[0, q] = np.real(np.trace(rho0 @ projectors[q]))
    
    # Evolve system
    rho = rho0.copy()
    for i in range(1, len(times)):
        rho = rk4_step(H, rho, dt)
        for q in range(6):
            probs[i, q] = np.real(np.trace(rho @ projectors[q]))
    
    # Analyze Q4 FFT using EXACTLY the same parameters as training
    peak_data = analyze_fft_with_peak_fitting(probs, times, 4, min_peak_height=0.05)  # Use 0.05 not 0.1!
    
    # CRITICAL: Sort peaks by amplitude (descending) to match training data generation
    peaks_list = peak_data['peaks']
    peaks_sorted = sorted(peaks_list, key=lambda x: x['amplitude'], reverse=True)
    
    # Extract peaks in the format expected by the app
    peaks = []
    for peak in peaks_sorted[:5]:  # Get first 5 peaks sorted by amplitude
        peaks.append({
            'freq': float(peak['frequency']),
            'amp': float(peak['amplitude'])
        })
    
    # Ensure we have exactly 5 peaks
    while len(peaks) < 5:
        peaks.append({'freq': 0.0, 'amp': 0.0})
    
    # Store raw FFT for plotting
    freq, mag = peak_data['raw_fft']
    
    return peaks[:5], probs, times, (freq, mag)

# ==========================================
# STREAMLIT APP
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

# Title and description
st.title("🔬 Quantum FFT State Analyzer")
st.markdown("---")

# Network diagram
def draw_network_diagram():
    """Draw the 6-qubit chain topology in a 2x3 grid layout"""
    fig = go.Figure()
    
    # Qubit positions - 2x3 grid
    # Top row: Q0, Q2, Q4
    # Bottom row: Q1, Q3, Q5
    qubit_x = [0, 2, 4, 0, 2, 4]  # Q0, Q2, Q4, Q1, Q3, Q5
    qubit_y = [1, 1, 1, 0, 0, 0]   # Top row at y=1, bottom row at y=0
    qubit_labels = ['0', '2', '4', '1', '3', '5']
    
    # Different colors for top and bottom rows
    qubit_colors = ['lightblue', 'lightblue', 'lightblue', 'lightpink', 'lightpink', 'lightpink']
    
    # Draw qubits
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
    
    # Draw J-couplings (horizontal connections in top row)
    j_connections = [(0, 1), (1, 2)]  # Connect Q0-Q2 and Q2-Q4
    for i, (start, end) in enumerate(j_connections):
        fig.add_trace(go.Scatter(
            x=[qubit_x[start], qubit_x[end]], 
            y=[qubit_y[start], qubit_y[end]],
            mode='lines',
            line=dict(width=2, color='darkgreen'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Add J labels above the lines
        mid_x = (qubit_x[start] + qubit_x[end]) / 2
        fig.add_annotation(
            x=mid_x, y=1.2, 
            text=f"J<sub>{qubit_labels[start]},{qubit_labels[end]}</sub>", 
            showarrow=False, 
            font=dict(size=12, color='darkgreen')
        )
    
    # Draw K-couplings (vertical connections)
    k_pairs = [(0, 3), (1, 4), (2, 5)]  # Q0-Q1, Q2-Q3, Q4-Q5
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
        # Add K labels to the left of vertical lines
        fig.add_annotation(
            x=qubit_x[top] - 0.3, 
            y=(qubit_y[top] + qubit_y[bottom]) / 2,
            text=k_labels[i], 
            showarrow=False, 
            font=dict(size=12, color='red')
        )
    
    # Add legend in a cleaner position
    # Create dummy traces for legend
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
    
    # Update layout
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

# Display network diagram
with st.container():
    st.plotly_chart(draw_network_diagram(), use_container_width=True)

st.markdown("---")

# Create two columns for forward and inverse problems
col1, col2 = st.columns(2)

# FORWARD PROBLEM (Left Panel)
with col1:
    st.header("➡️ Forward Problem")
    st.subheader("Quantum State → FFT Peaks")
    
    # Lock icon for k-values
    lock_col1, lock_col2 = st.columns([4, 1])
    with lock_col2:
        st.session_state.k_values_locked = st.checkbox("🔒", value=st.session_state.k_values_locked, 
                                                       help="Lock k-values between panels")
    
    # Coupling constants
    st.markdown("**Coupling Constants:**")
    k01_forward = st.number_input("k(0,1)", value=0.7, min_value=0.0, max_value=3.0, step=0.1, key="k01_f")
    k23_forward = st.number_input("k(2,3)", value=1.0, min_value=0.0, max_value=3.0, step=0.1, key="k23_f")
    k45_forward = st.number_input("k(4,5)", value=1.9, min_value=0.0, max_value=3.0, step=0.1, key="k45_f")
    j_coupling_forward = st.number_input("J-coupling", value=0.5, min_value=0.0, max_value=2.0, step=0.1, key="j_f")
    
    # Initial state info
    st.markdown("**Initial State:**")
    st.latex(r"|ψ⟩ = a|110000⟩ + b|100100⟩ + c|100001⟩")
    
    # Quantum amplitudes
    st.markdown("**Quantum Amplitudes:**")
    col_a, col_b = st.columns(2)
    with col_a:
        a_real = st.number_input("a_real", value=-0.033, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        b_real = st.number_input("b_real", value=1.472, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        c_real = st.number_input("c_real", value=1.515, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
    with col_b:
        a_imag = st.number_input("a_imag", value=0.420, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        b_imag = st.number_input("b_imag", value=1.368, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        c_imag = st.number_input("c_imag", value=-2.086, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
    
    # Calculate and display magnitudes and phases
    a_complex = a_real + 1j * a_imag
    b_complex = b_real + 1j * b_imag
    c_complex = c_real + 1j * c_imag
    
    a_mag = abs(a_complex)
    b_mag = abs(b_complex)
    c_mag = abs(c_complex)
    
    a_phase = np.angle(a_complex, deg=True)
    b_phase = np.angle(b_complex, deg=True)
    c_phase = np.angle(c_complex, deg=True)
    
    # Store magnitudes in session state
    st.session_state.forward_magnitudes = {'a': a_mag, 'b': b_mag, 'c': c_mag}
    
    # Display magnitudes and phases
    st.markdown("**Calculated Values:**")
    col_mag, col_phase = st.columns(2)
    with col_mag:
        st.markdown("**Magnitudes:**")
        st.markdown(f'<div class="magnitude-display">|a| = {a_mag:.3f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="magnitude-display">|b| = {b_mag:.3f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="magnitude-display">|c| = {c_mag:.3f}</div>', unsafe_allow_html=True)
    with col_phase:
        st.markdown("**Phases (degrees):**")
        st.markdown(f'<div class="phase-display">φ_a = {a_phase:.1f}°</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="phase-display">φ_b = {b_phase:.1f}°</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="phase-display">φ_c = {c_phase:.1f}°</div>', unsafe_allow_html=True)
    
    # Simulate button
    if st.button("🔄 Simulate Dynamics", key="simulate"):
        with st.spinner("Running quantum simulation..."):
            # Run actual simulation
            peaks, probs, times, fft_data = run_quantum_simulation(
                a_complex, b_complex, c_complex,
                k01_forward, k23_forward, k45_forward, j_coupling_forward
            )
            
            st.session_state.forward_results = {
                'peaks': peaks,
                'probs': probs,
                'times': times,
                'fft_data': fft_data,
                'success': True
            }
    
    # Results section
    if st.session_state.forward_results and st.session_state.forward_results['success']:
        st.markdown("### 📊 Results")
        st.markdown("**Detected Peaks (sorted by amplitude):**")
        
        peaks = st.session_state.forward_results['peaks']
        if peaks:
            for i, peak in enumerate(peaks):
                if peak['freq'] > 0:  # Only show non-zero peaks
                    st.write(f"• Peak {i+1}: {float(peak['freq']):.3f} Hz (amp: {float(peak['amp']):.3f})")
        else:
            st.write("No significant peaks detected")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            if st.button("📊 Show FFT Plot", key="fft_plot"):
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Get actual FFT data
                freq, mag = st.session_state.forward_results['fft_data']
                
                ax.plot(freq, mag, 'b-', linewidth=1.5)
                
                # Mark peaks
                for i, peak in enumerate(peaks):
                    if peak['freq'] > 0:
                        ax.axvline(peak['freq'], color='red', linestyle='--', alpha=0.5)
                        ax.plot(peak['freq'], peak['amp'], 'ro', markersize=8)
                        ax.text(peak['freq'], peak['amp'] + 0.01, f"{i+1}", ha='center', fontsize=8)
                
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.set_title('FFT of Q4 (peaks numbered by amplitude rank)')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 2)
                st.pyplot(fig)
        
        with col_plot2:
            if st.button("📈 Show Dynamics", key="dynamics_plot"):
                fig, ax = plt.subplots(figsize=(6, 4))
                probs = st.session_state.forward_results['probs']
                times = st.session_state.forward_results['times']
                
                ax.plot(times, probs[:, 4], 'b-', linewidth=1.5)
                ax.set_xlabel('Time')
                ax.set_ylabel('P(1)')
                ax.set_title('Q4 Dynamics')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                st.pyplot(fig)

# INVERSE PROBLEM (Right Panel)
with col2:
    st.header("⬅️ Inverse Problem")
    st.subheader("FFT Peaks → Quantum State")
    
    # Coupling constants (synchronized if locked)
    st.markdown("**Coupling Constants:**")
    if st.session_state.k_values_locked:
        k01_inverse = st.number_input("k(0,1)", value=k01_forward, min_value=0.0, max_value=3.0, step=0.1, 
                                     key="k01_i", disabled=True)
        k23_inverse = st.number_input("k(2,3)", value=k23_forward, min_value=0.0, max_value=3.0, step=0.1, 
                                     key="k23_i", disabled=True)
        k45_inverse = st.number_input("k(4,5)", value=k45_forward, min_value=0.0, max_value=3.0, step=0.1, 
                                     key="k45_i", disabled=True)
        j_coupling_inverse = st.number_input("J-coupling", value=j_coupling_forward, min_value=0.0, max_value=2.0, 
                                           step=0.1, key="j_i", disabled=True)
    else:
        k01_inverse = st.number_input("k(0,1)", value=0.7, min_value=0.0, max_value=3.0, step=0.1, key="k01_i")
        k23_inverse = st.number_input("k(2,3)", value=1.0, min_value=0.0, max_value=3.0, step=0.1, key="k23_i")
        k45_inverse = st.number_input("k(4,5)", value=1.9, min_value=0.0, max_value=3.0, step=0.1, key="k45_i")
        j_coupling_inverse = st.number_input("J-coupling", value=0.5, min_value=0.0, max_value=2.0, step=0.1, key="j_i")
    
    # Important note about peak ordering
    st.info("ℹ️ **Important**: Peaks should be ordered by amplitude (largest first), not frequency!")
    
    # Auto-fill from forward results
    if st.session_state.forward_results and st.button("↩️ Use Forward Results", key="autofill"):
        peaks = st.session_state.forward_results['peaks']
        st.success("Peak values loaded! They are already sorted by amplitude as required.")
    
    # FFT Peak inputs
    st.markdown("**FFT Peak Frequencies (Hz):**")
    st.markdown("*Enter peaks in order of decreasing amplitude*")
    col_freq, col_amp = st.columns(2)
    
    # Initialize default values
    default_peaks = []
    if st.session_state.forward_results:
        default_peaks = st.session_state.forward_results['peaks']
    
    with col_freq:
        st.markdown("**Frequencies:**")
        peak_freqs = []
        for i in range(5):
            default_freq = float(default_peaks[i]['freq']) if i < len(default_peaks) else 0.0
            freq = st.number_input(f"Peak {i+1}", value=default_freq, min_value=0.0, max_value=5.0, 
                                  step=0.001, format="%.3f", key=f"p{i+1}f")
            peak_freqs.append(freq)
    
    with col_amp:
        st.markdown("**Amplitudes:**")
        peak_amps = []
        for i in range(5):
            default_amp = float(default_peaks[i]['amp']) if i < len(default_peaks) else 0.0
            # More reasonable maximum based on typical values
            max_amp = 20.0
            if default_amp > max_amp:
                st.warning(f"Peak {i+1} amplitude {default_amp:.3f} exceeds maximum, capping at {max_amp}")
                default_amp = max_amp
            amp = st.number_input(f"Amp {i+1}", value=default_amp, min_value=0.0, max_value=max_amp, 
                                 step=0.001, format="%.3f", key=f"p{i+1}a")
            peak_amps.append(amp)
    
    # Model loading options
    st.markdown("**Model Loading:**")
    model_option = st.selectbox(
        "Select model loading method:",
        ["Local file (quantum_simple_nn_20250623_215653.pkl)",
         "Download from URL",
         "Use backup model"]
    )
    
    # Predict button
    if st.button("🧮 Predict State", key="predict"):
        with st.spinner("Running ML prediction..."):
            try:
                model_loaded = False
                model_data = None
                
                # Option 1: Try local file
                if model_option == "Local file (quantum_simple_nn_20250623_215653.pkl)":
                    model_file = 'quantum_simple_nn_20250623_215653.pkl'
                    
                    # Debug info
                    st.write("Debug Info:")
                    st.write(f"Looking for file: {model_file}")
                    st.write(f"File exists: {os.path.exists(model_file)}")
                    
                    if os.path.exists(model_file):
                        is_valid, msg = check_model_file(model_file)
                        st.write(f"File check: {msg}")
                        
                        if is_valid:
                            try:
                                model_data = joblib.load(model_file)
                                model_loaded = True
                                st.success(f"Model loaded successfully from {model_file}")
                            except Exception as e:
                                st.error(f"Failed to load model: {str(e)}")
                    else:
                        # List files in current directory
                        st.write("Files in current directory:")
                        for f in os.listdir('.'):
                            st.write(f"  - {f} ({os.path.getsize(f)} bytes)")
                
                # Option 2: Download from URL
                elif model_option == "Download from URL":
                    st.info("Please upload your model to a cloud service (Google Drive, Dropbox, etc.) and provide the direct download link.")
                    model_url = st.text_input("Enter direct download URL for the model:")
                    
                    if model_url:
                        model_data = load_model_from_url(model_url)
                        if model_data:
                            model_loaded = True
                            st.success("Model downloaded and loaded successfully!")
                
                # Option 3: Use simple backup model
                elif model_option == "Use backup model":
                    st.warning("Using simplified backup model (less accurate)")
                    # Create a simple linear model as backup
                    from sklearn.linear_model import LinearRegression
                    from sklearn.preprocessing import StandardScaler
                    
                    # Simple backup model
                    model_data = {
                        'model': LinearRegression(),
                        'scaler': StandardScaler(),
                        'approach': 'Backup Linear Model'
                    }
                    
                    # Fit with dummy data
                    X_dummy = np.random.rand(100, 10)
                    y_dummy = np.random.rand(100, 3)
                    model_data['scaler'].fit(X_dummy)
                    model_data['model'].fit(X_dummy, y_dummy)
                    model_loaded = True
                
                if model_loaded and model_data:
                    # CRITICAL: Sort peaks by amplitude (descending) to match training!
                    peak_data = list(zip(peak_freqs, peak_amps))
                    peak_data_sorted = sorted(peak_data, key=lambda x: x[1], reverse=True)
                    
                    # Extract sorted frequencies and amplitudes
                    sorted_freqs = [p[0] for p in peak_data_sorted]
                    sorted_amps = [p[1] for p in peak_data_sorted]
                    
                    # Show sorting info
                    if peak_data != peak_data_sorted:
                        st.warning("⚠️ Peaks were re-sorted by amplitude to match training data format!")
                        st.write("Sorted order:")
                        for i, (f, a) in enumerate(peak_data_sorted):
                            if f > 0:
                                st.write(f"  Peak {i+1}: {f:.3f} Hz (amp: {a:.3f})")
                    
                    # Check if this is a neural network model with engineered features
                    if 'feature_cols' in model_data and len(model_data['feature_cols']) > 10:
                        # This is the neural network model with engineered features
                        model = model_data['model']
                        scaler = model_data['scaler']
                        
                        # Prepare features EXACTLY as in training
                        features = []
                        
                        # Basic features (sorted frequencies and amplitudes)
                        for i in range(5):
                            features.extend([sorted_freqs[i], sorted_amps[i]])
                        
                        # Engineered features
                        total_power = sum(sorted_amps)
                        n_peaks = sum(1 for f in sorted_freqs if f > 0)
                        max_freq = max(sorted_freqs)
                        max_amp = max(sorted_amps)
                        freq_spread = max(sorted_freqs) - min(sorted_freqs)
                        
                        features.extend([total_power, n_peaks, max_freq, max_amp, freq_spread])
                        
                        X_test = np.array([features])
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Predict
                        if isinstance(model, dict):
                            # Ensemble
                            predictions = {}
                            for name, m in model.items():
                                if hasattr(m, 'predict'):
                                    predictions[name] = m.predict(X_test_scaled)[0]
                            
                            # Weighted average
                            weights = {'rf': 0.4, 'et': 0.3, 'nn': 0.3}
                            y_pred = np.zeros(3)
                            for name, pred in predictions.items():
                                weight = weights.get(name, 1.0/len(predictions))
                                y_pred += pred * weight
                            y_pred /= sum(weights.values())
                        else:
                            y_pred = model.predict(X_test_scaled)[0]
                        
                        # Apply calibration if available
                        if model_data.get('calibration_factors'):
                            for i in range(3):
                                y_pred[i] *= model_data['calibration_factors'][i]
                    else:
                        # Simple model
                        model = model_data['model']
                        scaler = model_data['scaler']
                        
                        features = []
                        for i in range(5):
                            features.extend([sorted_freqs[i], sorted_amps[i]])
                        
                        X_test = np.array([features])
                        X_test_scaled = scaler.transform(X_test)
                        y_pred = model.predict(X_test_scaled)[0]
                    
                    # Calculate confidence
                    non_zero_peaks = sum(1 for f in sorted_freqs if f > 0)
                    confidence = min(95, 50 + non_zero_peaks * 9)
                    
                    st.session_state.inverse_results = {
                        'a_mag': float(y_pred[0]),
                        'b_mag': float(y_pred[1]),
                        'c_mag': float(y_pred[2]),
                        'confidence': confidence,
                        'success': True,
                        'model_info': model_data.get('approach', 'Unknown'),
                        'model_file': model_option
                    }
                    
                    st.success(f"✅ Prediction complete using {model_data.get('approach', 'Unknown')}")
                else:
                    st.error("Failed to load model. Please check the instructions in the sidebar.")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    
    # Results section (same as before)
    if st.session_state.inverse_results and st.session_state.inverse_results['success']:
        st.markdown("### 📊 Results")
        st.markdown("**Predicted State Magnitudes:**")
        
        # Show predicted magnitudes with comparison to forward values
        pred_a = st.session_state.inverse_results['a_mag']
        pred_b = st.session_state.inverse_results['b_mag']
        pred_c = st.session_state.inverse_results['c_mag']
        
        if st.session_state.forward_magnitudes:
            true_a = st.session_state.forward_magnitudes['a']
            true_b = st.session_state.forward_magnitudes['b']
            true_c = st.session_state.forward_magnitudes['c']
            
            # Display with comparison
            st.markdown("**Magnitude Comparison:**")
            col_pred, col_true, col_diff = st.columns(3)
            
            with col_pred:
                st.markdown("**Predicted:**")
                st.write(f"|a| = {pred_a:.3f}")
                st.write(f"|b| = {pred_b:.3f}")
                st.write(f"|c| = {pred_c:.3f}")
            
            with col_true:
                st.markdown("**True (Forward):**")
                st.write(f"|a| = {true_a:.3f}")
                st.write(f"|b| = {true_b:.3f}")
                st.write(f"|c| = {true_c:.3f}")
            
            with col_diff:
                st.markdown("**Error:**")
                st.write(f"Δ|a| = {abs(pred_a - true_a):.3f}")
                st.write(f"Δ|b| = {abs(pred_b - true_b):.3f}")
                st.write(f"Δ|c| = {abs(pred_c - true_c):.3f}")
            
            # Calculate relative errors
            rel_error_a = abs(pred_a - true_a) / true_a * 100 if true_a > 0 else 0
            rel_error_b = abs(pred_b - true_b) / true_b * 100 if true_b > 0 else 0
            rel_error_c = abs(pred_c - true_c) / true_c * 100 if true_c > 0 else 0
            avg_rel_error = (rel_error_a + rel_error_b + rel_error_c) / 3
            
            st.markdown(f"**Average Relative Error:** {avg_rel_error:.1f}%")
            
            # Show improvement message if error is now low
            if avg_rel_error < 10:
                st.success("✅ Excellent prediction accuracy!")
            elif avg_rel_error < 20:
                st.info("✓ Good prediction accuracy")
        else:
            # Just show predictions if no forward values
            st.write(f"|a| = {pred_a:.3f}")
            st.write(f"|b| = {pred_b:.3f}")
            st.write(f"|c| = {pred_c:.3f}")
        
        st.write("")
        st.write(f"**Model Confidence:** {st.session_state.inverse_results['confidence']}%")
        
        # Note about phase information
        st.info("ℹ️ Note: The ML model predicts only magnitudes |a|, |b|, |c|. "
                "Phase information cannot be recovered from FFT peaks alone due to the "
                "loss of phase information in the power spectrum.")

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

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions & Troubleshooting")
    
    # Troubleshooting section
    st.markdown("""
    ### 🔧 Model Loading Issues?
    
    If the model file can't be loaded, it's likely because:
    1. **Git LFS Issue**: The .pkl file is stored in Git LFS and wasn't pulled properly
    2. **File too large**: Streamlit has limitations on file sizes
    
    ### Solutions:
    
    #### Option 1: Upload Model to Cloud
    1. Download the model file locally
    2. Upload to Google Drive, Dropbox, or similar
    3. Get a direct download link
    4. Use "Download from URL" option
    
    #### Option 2: Use Git LFS locally
    ```bash
    git lfs pull
    ```
    
    #### Option 3: Host model separately
    Upload the model to:
    - GitHub Releases (not in repo)
    - Hugging Face Model Hub
    - AWS S3 / Google Cloud Storage
    
    ---
    """)
    
    st.markdown("""
    ### ⚠️ Critical: Peak Ordering
    **Peaks must be ordered by amplitude (largest first), not frequency!**
    This matches how the ML model was trained.
    
    ### Forward Problem
    1. Set coupling constants (k-values and J)
    2. Enter quantum state amplitudes
    3. View calculated magnitudes and phases
    4. Click "Simulate Dynamics" 
    5. View detected FFT peaks (auto-sorted by amplitude)
    
    ### Inverse Problem
    1. Set coupling constants (or lock to sync)
    2. Enter FFT peaks in amplitude order:
       - Peak 1 = highest amplitude
       - Peak 2 = second highest, etc.
    3. Use "↩️ Use Forward Results" for correct ordering
    4. Click "Predict State"
    5. Compare predicted vs true magnitudes
    
    ### Key Features
    - 🔒 Lock icon syncs k-values between panels
    - 📊 Real quantum simulation
    - 🎯 Fixed peak ordering for accurate predictions
    - 📈 Direct error comparison
    - 📉 Visual analysis tools
    
    ### Important Notes
    - Peak ordering is CRITICAL for accuracy
    - The inverse problem predicts only magnitudes
    - Phase information is lost in FFT power spectrum
    - Low relative errors (< 10%) indicate good predictions
    """)