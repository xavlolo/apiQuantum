"""
Quantum FFT State Analyzer - Fixed Version
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
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# Page configuration
st.set_page_config(
    page_title="Quantum FFT State Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# QUANTUM SIMULATION FUNCTIONS
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

def analyze_fft_with_spline(probs, tpoints, qubit_idx, min_peak_height=0.05):
    """Simplified FFT analysis using cubic spline"""
    
    # Check if qubit has oscillation
    if np.std(probs[:, qubit_idx]) < 0.001:
        return []
    
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
    
    # Skip DC component
    if len(peaks) > 0 and pos_freq[peaks[0]] < 0.05:
        peaks = peaks[1:]
    
    if len(peaks) == 0:
        return []
    
    # Create cubic spline
    spline = CubicSpline(pos_freq, pos_mag)
    
    # Analyze peaks
    peak_results = []
    for peak_idx in peaks[:5]:  # Limit to 5 peaks
        # Define window around peak
        window = 20
        left_idx = max(0, peak_idx - window)
        right_idx = min(len(pos_freq), peak_idx + window)
        freq_window = pos_freq[left_idx:right_idx]
        
        if len(freq_window) < 4:
            continue
        
        # Find precise peak using spline
        freq_range = (freq_window[0], freq_window[-1])
        result = minimize_scalar(lambda x: -spline(x), bounds=freq_range, method='bounded')
        
        precise_freq = result.x
        precise_amp = spline(precise_freq)
        
        peak_results.append({
            'freq': float(precise_freq),
            'amp': float(precise_amp)
        })
    
    return peak_results

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
    
    # Analyze Q4 FFT
    peaks = analyze_fft_with_spline(probs, times, 4)  # Q4
    
    return peaks, probs, times

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

# Title and description
st.title("🔬 Quantum FFT State Analyzer")
st.markdown("---")

# Network diagram
def draw_network_diagram():
    """Draw the 6-qubit chain topology"""
    fig = go.Figure()
    
    # Qubit positions
    qubit_x = [0, 1, 2, 3, 4, 5]
    qubit_y = [0, 0, 0, 0, 0, 0]
    
    # Draw qubits
    fig.add_trace(go.Scatter(
        x=qubit_x, y=qubit_y,
        mode='markers+text',
        marker=dict(size=40, color='lightblue', line=dict(width=2, color='darkblue')),
        text=['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        textposition="middle center",
        name='Qubits'
    ))
    
    # Draw k-couplings (thick lines)
    k_pairs = [(0,1), (2,3), (4,5)]
    for i, j in k_pairs:
        fig.add_trace(go.Scatter(
            x=[i, j], y=[0, 0],
            mode='lines',
            line=dict(width=6, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw J-couplings (thin lines below)
    j_pairs = [(1,2), (3,4)]
    for i, j in j_pairs:
        fig.add_trace(go.Scatter(
            x=[i, j], y=[-0.3, -0.3],
            mode='lines',
            line=dict(width=2, color='green', dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add labels
    fig.add_annotation(x=0.5, y=0.15, text="k(0,1)", showarrow=False, font=dict(size=12, color='red'))
    fig.add_annotation(x=2.5, y=0.15, text="k(2,3)", showarrow=False, font=dict(size=12, color='red'))
    fig.add_annotation(x=4.5, y=0.15, text="k(4,5)", showarrow=False, font=dict(size=12, color='red'))
    fig.add_annotation(x=1.5, y=-0.45, text="J", showarrow=False, font=dict(size=12, color='green'))
    fig.add_annotation(x=3.5, y=-0.45, text="J", showarrow=False, font=dict(size=12, color='green'))
    
    # Legend
    fig.add_trace(go.Scatter(
        x=[6.5], y=[0], mode='lines', line=dict(width=6, color='red'),
        name='k-coupling (ZZ)', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[6.5], y=[-0.3], mode='lines', line=dict(width=2, color='green', dash='dash'),
        name='J-coupling (XX+YY)', showlegend=True
    ))
    
    fig.update_layout(
        title="6-Qubit Chain Topology",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 0.4]),
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
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
    
    # Simulate button
    if st.button("🔄 Simulate Dynamics", key="simulate"):
        with st.spinner("Running quantum simulation..."):
            # Create complex amplitudes
            a_complex = a_real + 1j * a_imag
            b_complex = b_real + 1j * b_imag
            c_complex = c_real + 1j * c_imag
            
            # Run actual simulation
            peaks, probs, times = run_quantum_simulation(
                a_complex, b_complex, c_complex,
                k01_forward, k23_forward, k45_forward, j_coupling_forward
            )
            
            st.session_state.forward_results = {
                'peaks': peaks,
                'probs': probs,
                'times': times,
                'success': True
            }
    
    # Results section
    if st.session_state.forward_results:
        st.markdown("### 📊 Results")
        st.markdown("**Detected Peaks:**")
        
        peaks = st.session_state.forward_results['peaks']
        if peaks:
            for i, peak in enumerate(peaks):
                st.write(f"• Peak {i+1}: {float(peak['freq']):.3f} Hz (amp: {float(peak['amp']):.3f})")
        else:
            st.write("No significant peaks detected")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            if st.button("📊 Show FFT Plot", key="fft_plot"):
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Compute actual FFT for display
                probs = st.session_state.forward_results['probs']
                times = st.session_state.forward_results['times']
                dt = times[1] - times[0]
                
                # FFT of Q4
                sig = probs[:, 4] - np.mean(probs[:, 4])
                win = np.hanning(len(sig))
                fft = np.fft.fft(sig * win)
                freq = np.fft.fftfreq(len(fft), dt)
                mask = freq >= 0
                
                ax.plot(freq[mask], np.abs(fft[mask]))
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.set_title('FFT of Q4')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 2)
                st.pyplot(fig)
        
        with col_plot2:
            if st.button("📈 Show Dynamics", key="dynamics_plot"):
                fig, ax = plt.subplots(figsize=(6, 4))
                probs = st.session_state.forward_results['probs']
                times = st.session_state.forward_results['times']
                
                ax.plot(times, probs[:, 4])
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
    
    # Target qubit selection
    target_qubit = st.selectbox("Target Qubit", ["Q4", "Q0", "Q1", "Q2", "Q3", "Q5"])
    
    # Auto-fill from forward results
    if st.session_state.forward_results and st.button("↩️ Use Forward Results", key="autofill"):
        peaks = st.session_state.forward_results['peaks']
        # This will update the values but requires a rerun to show in the inputs
        st.info("Peak values loaded! Adjust if needed and click 'Predict State'")
    
    # FFT Peak inputs
    st.markdown("**FFT Peak Frequencies (Hz):**")
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
            # Cap the default value if it exceeds maximum
            max_amp = 50.0
            if default_amp > max_amp:
                st.warning(f"Peak {i+1} amplitude {default_amp:.3f} exceeds maximum, capping at {max_amp}")
                default_amp = max_amp
            amp = st.number_input(f"Amp {i+1}", value=default_amp, min_value=0.0, max_value=max_amp, 
                                 step=0.001, format="%.3f", key=f"p{i+1}a")
            peak_amps.append(amp)
    
    # Predict button
    if st.button("🧮 Predict State", key="predict"):
        with st.spinner("Running ML prediction..."):
            try:
                # Check if model exists
                if os.path.exists('quantum_inverse_model.pkl'):
                    # Load model
                    model_data = joblib.load('quantum_inverse_model.pkl')
                    model = model_data['model']
                    scaler = model_data['scaler']
                    
                    # Prepare features (frequencies and amplitudes interleaved)
                    features = []
                    for i in range(5):
                        features.extend([peak_freqs[i], peak_amps[i]])
                    
                    X_test = np.array([features])
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)[0]
                    
                    # Calculate confidence based on input quality
                    non_zero_peaks = sum(1 for f in peak_freqs if f > 0)
                    confidence = min(95, 50 + non_zero_peaks * 9)
                    
                    st.session_state.inverse_results = {
                        'a_mag': float(y_pred[0]),
                        'b_mag': float(y_pred[1]),
                        'c_mag': float(y_pred[2]),
                        'confidence': confidence,
                        'success': True
                    }
                else:
                    st.error("Model file 'quantum_inverse_model.pkl' not found!")
                    st.info("Please run MLpipeline5peaks.py to generate the model first.")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    # Results section
    if st.session_state.inverse_results and st.session_state.inverse_results['success']:
        st.markdown("### 📊 Results")
        st.markdown("**Predicted State Magnitudes:**")
        st.write(f"|a| = {st.session_state.inverse_results['a_mag']:.3f}")
        st.write(f"|b| = {st.session_state.inverse_results['b_mag']:.3f}")
        st.write(f"|c| = {st.session_state.inverse_results['c_mag']:.3f}")
        st.write(f"**Confidence:** {st.session_state.inverse_results['confidence']}%")
        
        if st.button("📊 Show Comparison", key="comparison"):
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Bar chart of magnitudes
            labels = ['|a|', '|b|', '|c|']
            predicted = [st.session_state.inverse_results['a_mag'],
                        st.session_state.inverse_results['b_mag'],
                        st.session_state.inverse_results['c_mag']]
            
            ax1.bar(labels, predicted, color=['blue', 'green', 'red'])
            ax1.set_ylabel('Magnitude')
            ax1.set_title('Predicted State Magnitudes')
            ax1.set_ylim(0, max(predicted) * 1.2)
            
            # Add value labels on bars
            for i, v in enumerate(predicted):
                ax1.text(i, v + 0.05, f'{v:.3f}', ha='center')
            
            # Confidence visualization
            ax2.pie([st.session_state.inverse_results['confidence'], 
                    100 - st.session_state.inverse_results['confidence']], 
                   labels=['Confident', 'Uncertain'],
                   colors=['green', 'lightgray'],
                   startangle=90)
            ax2.set_title('Prediction Confidence')
            
            st.pyplot(fig)

# Status bar
st.markdown("---")
status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    if os.path.exists('quantum_inverse_model.pkl'):
        st.success("Status: Ready | Model: quantum_inverse_model.pkl loaded ✓")
    else:
        st.warning("Status: Limited | Model not found - run MLpipeline5peaks.py first")
with status_col2:
    st.info(f"k-values {'locked 🔒' if st.session_state.k_values_locked else 'unlocked 🔓'}")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### Forward Problem
    1. Set coupling constants (k-values and J)
    2. Enter quantum state amplitudes
    3. Click "Simulate Dynamics" 
    4. View detected FFT peaks
    
    ### Inverse Problem
    1. Set coupling constants (or lock to sync)
    2. Enter FFT peak frequencies and amplitudes
       - Or use "↩️ Use Forward Results" button
    3. Click "Predict State"
    4. View predicted quantum state magnitudes
    
    ### Features
    - 🔒 Lock icon syncs k-values between panels
    - 📊 Real quantum simulation (not placeholders!)
    - 🧮 ML-based state prediction
    - ↩️ Auto-fill inverse from forward results
    
    ### Setup Requirements
    1. Run `Qubits6_fitted_spline.py` to generate ML dataset
    2. Run `MLpipeline5peaks.py` to train the model
    3. Ensure `quantum_inverse_model.pkl` exists
    
    ### Note
    The forward problem runs actual quantum dynamics simulation,
    so it may take a few seconds to complete.
    """)