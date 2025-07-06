"""
Quantum FFT State Analyzer - 10-Qubit Version with Dynamic k-Couplings
Forward problem only: Quantum State → FFT Peaks
Now with flexible k-coupling selection and Perfect State Transfer (PST) options
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import product
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import time
import pandas as pd
from datetime import datetime
from quantum_report_generator import generate_quantum_report


# Page configuration
st.set_page_config(
    page_title="Dynamic 10-Qubit Quantum FFT Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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
    .magnitude-display {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    .progress-info {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .simulation-params {
        background-color: #fffbf0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    .pst-info {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .j-value-display {
        background-color: #f5f5f5;
        padding: 8px;
        border-radius: 4px;
        margin: 3px 0;
        font-family: monospace;
    }
    .k-coupling-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 10px 0;
    }
    .state-display {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# PST CALCULATION FUNCTIONS
# ==========================================

def calculate_pst_couplings(N, J_max):
    """Calculate Perfect State Transfer coupling strengths"""
    if N % 2 == 0:
        J_0 = 2 * J_max / N
        chain_type = "even"
    else:
        J_0 = 2 * J_max / N * np.sqrt(1 - 1/N**2)
        chain_type = "odd"
    
    couplings = []
    for i in range(1, N):
        J_i = J_0 * np.sqrt(i * (N - i))
        couplings.append(J_i)
    
    return couplings, J_0, chain_type

# ==========================================
# QUANTUM SIMULATION FUNCTIONS
# ==========================================

@st.cache_data
def get_full_basis_10qubits():
    basis_states = list(product([0, 1], repeat=10))
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
def generate_hamiltonian_10qubits_dynamic(_basis_states, _state_to_idx, _k_pattern, j_couplings):
    """Generate Hamiltonian with dynamic k-coupling pattern"""
    n = len(_basis_states)
    H = np.zeros((n, n), complex)
    
    # Convert k_pattern to regular dict for hashing
    k_pattern = dict(_k_pattern)
    
    # Diagonal elements (ZZ interactions)
    for idx, st in enumerate(_basis_states):
        H[idx, idx] = calculate_zz_energy(st, k_pattern)

    # Off-diagonal elements (XX+YY interactions) - horizontal chain only
    j_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
    for idx_pair, (i, j) in enumerate(j_pairs):
        J_coupling = j_couplings[idx_pair]
        for idx, st in enumerate(_basis_states):
            if st[i] != st[j]:
                flipped = list(st)
                flipped[i], flipped[j] = st[j], st[i]
                idx2 = _state_to_idx[tuple(flipped)]
                H[idx, idx2] += J_coupling
                H[idx2, idx] += J_coupling

    evals, evecs = np.linalg.eigh(H)
    return H, evals, evecs

def analyze_fft_with_peak_fitting(probs, tpoints, qubit_idx, min_peak_height=0.05):
    """FFT analysis using CubicSpline"""
    
    if np.std(probs[:, qubit_idx]) < 0.001:
        empty_freq = np.linspace(0, 1, 100)
        empty_mag = np.zeros_like(empty_freq)
        return {
            'raw_fft': (empty_freq, empty_mag),
            'peaks': [],
            'fitted_spectrum': empty_mag
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
    peaks, _ = find_peaks(pos_mag, height=threshold, distance=5)
    
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
        
        fitted_peaks.append({
            'frequency': precise_freq,
            'amplitude': precise_amp,
            'width': width,
            'Q_factor': Q_factor,
            'fit_success': True
        })
    
    fitted_peaks.sort(key=lambda x: x['amplitude'], reverse=True)
    
    while len(fitted_peaks) < 5:
        fitted_peaks.append({
            'frequency': 0.0,
            'amplitude': 0.0,
            'width': 0.0,
            'Q_factor': 0.0,
            'fit_success': False
        })
    
    if len(fitted_peaks) > 5:
        fitted_peaks = fitted_peaks[:5]
    
    return {
        'raw_fft': (pos_freq, pos_mag),
        'peaks': fitted_peaks,
        'fitted_spectrum': fitted_spectrum
    }

def run_quantum_simulation_dynamic(a_complex, b_complex, c_complex, 
                                 k_pattern, j_couplings,
                                 dt, t_max, progress_bar=None, status_text=None):
    """Run simulation with dynamic k-coupling pattern"""
    
    sim_start_time = time.time()
    
    if progress_bar:
        progress_bar.progress(0.05)
        status_text.text('🔧 Initializing quantum system...')
    
    basis_states, state_to_idx = get_full_basis_10qubits()
    projectors = [np.diag([s[q] for s in basis_states]) for q in range(10)]
    
    # Extract the connected bottom qubits from k_pattern (always 7,8,9)
    bottom_qubits = sorted([pair[1] for pair in k_pattern.keys()])
    
    # Dynamic initial states - always include qubit 0 + one of the bottom qubits
    state_A = tuple(1 if i in [0, bottom_qubits[0]] else 0 for i in range(10))
    state_B = tuple(1 if i in [0, bottom_qubits[1]] else 0 for i in range(10))
    state_C = tuple(1 if i in [0, bottom_qubits[2]] else 0 for i in range(10))
    
    superposition_dict = {
        state_A: a_complex,
        state_B: b_complex,
        state_C: c_complex
    }
    
    if progress_bar:
        progress_bar.progress(0.10)
        status_text.text('🌐 Creating quantum superposition...')
    
    psi0, rho0 = initialize_superposition(basis_states, state_to_idx, superposition_dict)
    
    if progress_bar:
        progress_bar.progress(0.15)
        status_text.text('🏗️ Building 1024×1024 Hamiltonian matrix...')
    
    H, evals, evecs = generate_hamiltonian_10qubits_dynamic(
        basis_states, state_to_idx, k_pattern, j_couplings
    )
    
    times = np.arange(0, t_max + dt, dt)
    n_steps = len(times)
    probs = np.zeros((n_steps, 10))
    
    for q in range(10):
        probs[0, q] = np.real(np.trace(rho0 @ projectors[q]))
    
    if progress_bar:
        progress_bar.progress(0.20)
        status_text.text(f'⏱️ Starting time evolution ({n_steps} steps)...')
    
    rho = rho0.copy()
    update_frequency = max(1, n_steps // 100)
    
    for i in range(1, n_steps):
        rho = rk4_step(H, rho, dt)
        for q in range(10):
            probs[i, q] = np.real(np.trace(rho @ projectors[q]))
        
        if progress_bar and i % update_frequency == 0:
            progress = 0.20 + (i / n_steps) * 0.70
            progress_bar.progress(progress)
            percent_complete = (i / n_steps) * 100
            elapsed = time.time() - sim_start_time
            time_remaining = (elapsed / i) * (n_steps - i) if i > 0 else 0
            status_text.text(f'⏱️ Time evolution: Step {i}/{n_steps} ({percent_complete:.0f}%) - Est. {time_remaining:.1f}s remaining')
    
    if progress_bar:
        progress_bar.progress(0.92)
        status_text.text('📊 Performing FFT analysis...')
    
    peak_data = analyze_fft_with_peak_fitting(probs, times, 6, min_peak_height=0.05)
    
    if progress_bar:
        progress_bar.progress(0.97)
        status_text.text('🔍 Extracting peak frequencies...')
    
    peaks_list = peak_data['peaks']
    
    peaks = []
    for peak in peaks_list[:5]:
        peaks.append({
            'freq': float(peak['frequency']),
            'amp': float(peak['amplitude'])
        })
    
    while len(peaks) < 5:
        peaks.append({'freq': 0.0, 'amp': 0.0})
    
    freq, mag = peak_data['raw_fft']
    
    if progress_bar:
        progress_bar.progress(1.0)
        status_text.text('✅ Simulation complete!')
    
    return peaks[:5], probs, times, (freq, mag)

def draw_network_diagram_dynamic(k_pattern):
    """Draw the 10-qubit topology with dynamic k-couplings"""
    fig = go.Figure()
    
    qubit_x = [0, 2, 4, 6, 8, 10, 12,
               0, 2, 4]
    qubit_y = [1, 1, 1, 1, 1, 1, 1,
               0, 0, 0]
    qubit_labels = ['0', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9']
    
    qubit_colors = ['lightblue']*7 + ['lightpink']*3
    
    fig.add_trace(go.Scatter(
        x=qubit_x, y=qubit_y,
        mode='markers+text',
        marker=dict(size=50, color=qubit_colors, line=dict(width=2, color='darkgray')),
        text=qubit_labels,
        textposition="middle center",
        textfont=dict(size=14, color='black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # J-couplings (horizontal chain - fixed)
    j_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    for i, j in j_connections:
        fig.add_trace(go.Scatter(
            x=[qubit_x[i], qubit_x[j]], 
            y=[qubit_y[i], qubit_y[j]],
            mode='lines',
            line=dict(width=2, color='darkgreen'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Dynamic k-couplings (vertical connections)
    for (top, bottom) in k_pattern.keys():
        fig.add_trace(go.Scatter(
            x=[qubit_x[top], qubit_x[bottom]], 
            y=[qubit_y[top], qubit_y[bottom]],
            mode='lines',
            line=dict(width=4, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Add label
        fig.add_annotation(
            x=qubit_x[top] - 0.3, 
            y=0.5,
            text=f"k({top},{bottom})", 
            showarrow=False, 
            font=dict(size=10, color='red')
        )
    
    # Legend
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
    
    # Monitor qubit annotation
    fig.add_annotation(
        x=qubit_x[6], y=1.5,
        text="Monitor Q6",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="blue",
        ax=0,
        ay=-30
    )
    
    fig.update_layout(
        title=dict(
            text="10-Qubit Network Topology (Dynamic k-Couplings)",
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
            range=[-1, 13]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-0.5, 2]
        ),
        height=350,
        margin=dict(l=50, r=150, t=50, b=50),
        plot_bgcolor='white'
    )
    
    return fig

# ==========================================
# MAIN APP
# ==========================================

st.title("🔬 Dynamic 10-Qubit Quantum FFT Analyzer")
st.markdown("Forward Problem: Quantum State → FFT Peaks (Monitoring Q6)")

# ==========================================
# K-COUPLING SELECTION UI
# ==========================================

st.markdown("---")
st.markdown("### 🔗 k-Coupling Configuration")
st.markdown('<div class="k-coupling-box">', unsafe_allow_html=True)
st.markdown("**Select which top qubits (0-6) connect to bottom qubits (7,8,9):**")

col_k1, col_k2, col_k3 = st.columns(3)
with col_k1:
    top_qubit_1 = st.selectbox("Top qubit → Q7", options=[0,1,2,3,4,5,6], index=0, key="k1")
with col_k2:
    top_qubit_2 = st.selectbox("Top qubit → Q8", options=[0,1,2,3,4,5,6], index=1, key="k2")  
with col_k3:
    top_qubit_3 = st.selectbox("Top qubit → Q9", options=[0,1,2,3,4,5,6], index=2, key="k3")

# Validation
selected_qubits = [top_qubit_1, top_qubit_2, top_qubit_3]
if len(set(selected_qubits)) != 3:
    st.error("⚠️ Please select 3 different qubits!")
    st.stop()

st.success(f"✅ Selected connections: Q{top_qubit_1}↔Q7, Q{top_qubit_2}↔Q8, Q{top_qubit_3}↔Q9")
st.markdown("</div>", unsafe_allow_html=True)

# Build k_pattern based on selections
k_pattern = {
    (top_qubit_1, 7): 0.7,  # These will be updated by sliders below
    (top_qubit_2, 8): 1.0,
    (top_qubit_3, 9): 1.9
}

# Show dynamic network diagram
with st.container():
    st.plotly_chart(draw_network_diagram_dynamic(k_pattern), use_container_width=True)

# Add system info
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown('<div class="progress-info">💻 <b>System Size:</b><br>10 qubits = 1024 basis states</div>', unsafe_allow_html=True)
with col_info2:
    st.markdown('<div class="progress-info">⏱️ <b>Typical Runtime:</b><br>15-30 seconds</div>', unsafe_allow_html=True)
with col_info3:
    st.markdown('<div class="progress-info">🎯 <b>Monitoring:</b><br>Qubit Q6 (last in chain)</div>', unsafe_allow_html=True)

st.markdown("---")

# Input parameters
col1, col2 = st.columns(2)

with col1:
    st.header("System Parameters")
    
    # k-Coupling constants with dynamic labels
    st.markdown("**k-Coupling Strengths (ZZ):**")
    k_val_1 = st.slider(f"k({top_qubit_1},7)", 0.0, 3.0, 0.7, 0.1)
    k_val_2 = st.slider(f"k({top_qubit_2},8)", 0.0, 3.0, 1.0, 0.1)
    k_val_3 = st.slider(f"k({top_qubit_3},9)", 0.0, 3.0, 1.9, 0.1)
    
    # Update k_pattern with actual values
    k_pattern = {
        (top_qubit_1, 7): k_val_1,
        (top_qubit_2, 8): k_val_2,
        (top_qubit_3, 9): k_val_3
    }
    
    # J-Coupling mode selection
    st.markdown("---")
    st.markdown("**J-Coupling Mode (XX+YY):**")
    j_mode = st.radio("Select J-coupling mode:", 
                      ["Uniform J", "Custom J", "PST (Perfect State Transfer)"],
                      index=0)
    
    if j_mode == "Uniform J":
        j_uniform = st.slider("J (all horizontal)", 0.0, 2.0, 0.5, 0.1)
        j_couplings = [j_uniform] * 6
        
    elif j_mode == "Custom J":
        st.markdown("**Individual J-couplings:**")
        j_couplings = []
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            j_couplings.append(st.number_input("J(0,1)", value=0.5, min_value=0.0, max_value=3.0, step=0.01, format="%.3f"))
            j_couplings.append(st.number_input("J(1,2)", value=0.5, min_value=0.0, max_value=3.0, step=0.01, format="%.3f"))
            j_couplings.append(st.number_input("J(2,3)", value=0.5, min_value=0.0, max_value=3.0, step=0.01, format="%.3f"))
        with col_j2:
            j_couplings.append(st.number_input("J(3,4)", value=0.5, min_value=0.0, max_value=3.0, step=0.01, format="%.3f"))
            j_couplings.append(st.number_input("J(4,5)", value=0.5, min_value=0.0, max_value=3.0, step=0.01, format="%.3f"))
            j_couplings.append(st.number_input("J(5,6)", value=0.5, min_value=0.0, max_value=3.0, step=0.01, format="%.3f"))
    
    else:  # PST mode
        st.markdown('<div class="pst-info">', unsafe_allow_html=True)
        st.markdown("**PST Configuration:**")
        
        pst_n = st.number_input("PST Chain Length N", 
                               value=7, min_value=2, max_value=20, step=1,
                               help="Calculate PST for N-site chain. First 6 values will be used.")
        
        j_max_pst = st.slider("J_max for PST", 0.0, 3.0, 1.0, 0.1,
                             help="Maximum coupling strength for PST calculation")
        
        # Calculate PST values
        pst_values, j0_value, chain_type = calculate_pst_couplings(pst_n, j_max_pst)
        
        # Display formula and J₀
        st.markdown(f"**Chain type:** {chain_type} (N={pst_n})")
        if chain_type == "even":
            st.latex(f"J_0 = \\frac{{2 J_{{max}}}}{{N}} = \\frac{{2 \\times {j_max_pst:.2f}}}{{{pst_n}}} = {j0_value:.4f}")
        else:
            st.latex(f"J_0 = \\frac{{2 J_{{max}}}}{{N}} \\sqrt{{1 - \\frac{{1}}{{N^2}}}} = {j0_value:.4f}")
        
        st.markdown("**Calculated PST values:**")
        
        # Show all calculated values
        for i in range(len(pst_values)):
            if i < 6:
                st.markdown(f'<div class="j-value-display">J({i},{i+1}) = {j0_value:.4f} × √({i+1}×{pst_n-i-1}) = <b>{pst_values[i]:.4f}</b></div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="j-value-display" style="opacity: 0.5;">J({i},{i+1}) = {pst_values[i]:.4f} (not used)</div>', 
                          unsafe_allow_html=True)
        
        # Use first 6 values for our chain
        if len(pst_values) >= 6:
            j_couplings = pst_values[:6]
            if len(pst_values) > 6:
                st.info(f"Using first 6 values from {pst_n}-site PST calculation")
        else:
            # Pad with zeros if PST chain is shorter than 6
            j_couplings = pst_values + [0.0] * (6 - len(pst_values))
            st.warning(f"⚠️ PST chain (N={pst_n}) is shorter than physical chain (7 sites). Padding with zeros.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Simulation parameters
    st.markdown("---")
    st.markdown('<div class="simulation-params">', unsafe_allow_html=True)
    st.markdown("**⚙️ Simulation Parameters:**")
    
    col_dt, col_tmax = st.columns(2)
    with col_dt:
        dt = st.number_input("Time step (dt)", 
                            value=0.1, 
                            min_value=0.001, 
                            max_value=1.0, 
                            step=0.001,
                            format="%.3f",
                            help="Smaller dt gives more accurate results but takes longer")
    with col_tmax:
        t_max = st.number_input("Max time", 
                               value=10.0, 
                               min_value=10.0, 
                               max_value=200.0, 
                               step=5.0,
                               format="%.1f",
                               help="Longer time captures more oscillation periods")
    
    n_steps = int(t_max / dt) + 1
    st.caption(f"📊 Number of time steps: {n_steps}")
    if n_steps > 10000:
        st.warning("⚠️ Large number of steps may take significant time!")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.header("Initial State")
    
    # Dynamic state display
    st.markdown('<div class="state-display">', unsafe_allow_html=True)
    st.markdown("**Dynamic Quantum State Configuration:**")
    
    # Generate state strings for display - use BOTTOM qubits (7,8,9)
    bottom_qubits = sorted([pair[1] for pair in k_pattern.keys()])
    
    state_A = tuple(1 if i in [0, bottom_qubits[0]] else 0 for i in range(10))
    state_B = tuple(1 if i in [0, bottom_qubits[1]] else 0 for i in range(10))
    state_C = tuple(1 if i in [0, bottom_qubits[2]] else 0 for i in range(10))
    
    state_str_A = ''.join(map(str, state_A))
    state_str_B = ''.join(map(str, state_B))
    state_str_C = ''.join(map(str, state_C))
    
    st.latex(f"|ψ⟩ = a|{state_str_A}⟩ + b|{state_str_B}⟩ + c|{state_str_C}⟩")
    
    st.markdown("**State Interpretation:**")
    
    # Bottom qubits are always 7,8,9 - show which ones are excited with qubit 0
    bottom_qubits = sorted([pair[1] for pair in k_pattern.keys()])
    
    st.markdown(f"- **a**: Qubits 0 and {bottom_qubits[0]} excited")
    st.markdown(f"- **b**: Qubits 0 and {bottom_qubits[1]} excited")
    st.markdown(f"- **c**: Qubits 0 and {bottom_qubits[2]} excited")
    st.markdown("- Qubit 0 (top) + bottom qubits (7,8,9) pattern")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Quantum amplitudes
    st.markdown("---")
    st.markdown("**Complex Amplitudes:**")
    col_real, col_imag = st.columns(2)
    with col_real:
        st.markdown("**Real parts:**")
        a_real = st.number_input("a_real", value=0.148, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        b_real = st.number_input("b_real", value=-2.004, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        c_real = st.number_input("c_real", value=1.573, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
    with col_imag:
        st.markdown("**Imaginary parts:**")
        a_imag = st.number_input("a_imag", value=2.026, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        b_imag = st.number_input("b_imag", value=0.294, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
        c_imag = st.number_input("c_imag", value=-2.555, min_value=-3.0, max_value=3.0, step=0.001, format="%.3f")
    
    # Display current coupling summary
    if j_mode in ["Custom J", "PST (Perfect State Transfer)"]:
        st.markdown("---")
        st.markdown("**Active Coupling Summary:**")
        
        # k-couplings
        k_summary = pd.DataFrame({
            'k-Coupling': [f'k({top_qubit_1},7)', f'k({top_qubit_2},8)', f'k({top_qubit_3},9)'],
            'Value': [f"{k_val_1:.3f}", f"{k_val_2:.3f}", f"{k_val_3:.3f}"]
        })
        st.dataframe(k_summary, hide_index=True, use_container_width=True)
        
        # J-couplings preview
        with st.expander("J-coupling details"):
            j_summary = pd.DataFrame({
                'J-Coupling': ['J(0,1)', 'J(1,2)', 'J(2,3)', 'J(3,4)', 'J(4,5)', 'J(5,6)'],
                'Value': [f"{j:.4f}" for j in j_couplings]
            })
            st.dataframe(j_summary, hide_index=True, use_container_width=True)

# Calculate complex amplitudes
a_complex = a_real + 1j * a_imag
b_complex = b_real + 1j * b_imag
c_complex = c_real + 1j * c_imag

# Display magnitudes
col_mag1, col_mag2, col_mag3 = st.columns(3)
with col_mag1:
    st.markdown(f'<div class="magnitude-display">|a| = {abs(a_complex):.3f}</div>', unsafe_allow_html=True)
with col_mag2:
    st.markdown(f'<div class="magnitude-display">|b| = {abs(b_complex):.3f}</div>', unsafe_allow_html=True)
with col_mag3:
    st.markdown(f'<div class="magnitude-display">|c| = {abs(c_complex):.3f}</div>', unsafe_allow_html=True)

# Replace the "Run simulation" button section and everything after it with this:

# Run simulation
if st.button("🚀 Run Simulation", use_container_width=True):
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    start_time = time.time()
    
    try:
        peaks, probs, times, fft_data = run_quantum_simulation_dynamic(
            a_complex, b_complex, c_complex,
            k_pattern, j_couplings,
            dt, t_max,
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        elapsed_time = time.time() - start_time
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state for reactive plotting
        st.session_state.simulation_results = {
            'peaks': peaks,
            'probs': probs,
            'times': times,
            'fft_data': fft_data,
            'elapsed_time': elapsed_time
        }
        
        st.success(f"✅ Simulation complete! Total time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Simulation failed: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())

# Results section - this runs OUTSIDE the button block so it's reactive
if 'simulation_results' in st.session_state:
    results = st.session_state.simulation_results
    probs = results['probs']
    times = results['times']
    peaks = results['peaks']
    fft_data = results['fft_data']
    elapsed_time = results['elapsed_time']
    
    # Results
    st.markdown("---")
    st.header("📊 Results")
    
    # Add probe selection for FFT analysis
    st.markdown("### 🎯 Probe Selection for FFT Analysis")
    st.info("💡 **Interactive Selection**: Change the probe qubit below and all FFT analysis will update automatically - no need to re-run simulation!")
    probe_qubit = st.selectbox(
        "Select which qubit to analyze with FFT:",
        options=list(range(7)),  # Top qubits 0-6
        index=6,  # Default to Q6
        format_func=lambda x: f"Q{x} (Top Layer)",
        help="FFT analysis will update immediately when you change this selection"
    )
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.subheader("All Top Qubits Dynamics")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all top qubits (0-6) with different colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        for i in range(7):
            if i == probe_qubit:
                # Highlight the selected probe qubit
                ax.plot(times, probs[:, i], color=colors[i], linewidth=3, 
                       label=f'Q{i} (PROBE)', alpha=1.0, zorder=10)
            else:
                # Regular qubits
                ax.plot(times, probs[:, i], color=colors[i], linewidth=1.5, 
                       label=f'Q{i}', alpha=0.7, zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('P(1) - Occupation Probability')
        ax.set_title('All Top Qubits (Q0-Q6) Probability vs Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics for all qubits
        st.markdown("**Qubit Statistics:**")
        stats_data = []
        for i in range(7):
            avg_prob = np.mean(probs[:, i])
            osc_amp = np.max(probs[:, i]) - np.min(probs[:, i])
            max_prob = np.max(probs[:, i])
            qubit_label = f'Q{i} (PROBE)' if i == probe_qubit else f'Q{i}'
            stats_data.append({
                'Qubit': qubit_label,
                'Avg P(1)': f'{avg_prob:.3f}',
                'Max P(1)': f'{max_prob:.3f}',
                'Oscillation': f'{osc_amp:.3f}'
            })
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, hide_index=True, use_container_width=True)
    
    with col_res2:
        st.subheader(f"FFT Analysis - Q{probe_qubit} (Probe)")
        
        # This section will automatically update when probe_qubit changes
        # Perform FFT on selected probe qubit
        probe_signal = probs[:, probe_qubit]
        fft_result = np.fft.fft(probe_signal)
        freqs = np.fft.fftfreq(len(times), times[1] - times[0])
        
        # Keep only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_result[:len(freqs)//2])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(positive_freqs, magnitude, 'b-', linewidth=1.5)
        
        # Find and mark peaks
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1)
        
        for i, peak_idx in enumerate(peaks_idx[:5]):  # Show top 5 peaks
            freq_val = positive_freqs[peak_idx]
            mag_val = magnitude[peak_idx]
            if freq_val > 0:  # Skip DC component
                ax.axvline(freq_val, color='red', linestyle='--', alpha=0.5)
                ax.plot(freq_val, mag_val, 'ro', markersize=8)
                ax.text(freq_val, mag_val + np.max(magnitude)*0.02, 
                       f"{i+1}", ha='center', fontsize=8)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'FFT of Q{probe_qubit} (peaks numbered by prominence)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2)
        st.pyplot(fig)
        
        # Peak information for probe qubit (will update automatically)
        st.markdown(f"**Q{probe_qubit} Probe Statistics:**")
        probe_avg = np.mean(probe_signal)
        probe_osc = np.max(probe_signal) - np.min(probe_signal)
        probe_max = np.max(probe_signal)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Average P(1)", f"{probe_avg:.3f}")
            st.metric("Max P(1)", f"{probe_max:.3f}")
        with col_stat2:
            st.metric("Oscillation Amp", f"{probe_osc:.3f}")
            st.metric("Probe Qubit", f"Q{probe_qubit}")
    
    # Enhanced peak detection and analysis (will update when probe changes)
    st.markdown("---")
    st.subheader(f"🔍 Detailed Peak Analysis - Q{probe_qubit}")
    
    # More sophisticated peak detection - recalculate for current probe
    current_probe_signal = probs[:, probe_qubit]
    current_fft_result = np.fft.fft(current_probe_signal)
    current_freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    current_positive_freqs = current_freqs[:len(current_freqs)//2]
    current_magnitude = np.abs(current_fft_result[:len(current_freqs)//2])
    
    # Find peaks with better parameters
    current_peaks_idx, current_properties = find_peaks(
        current_magnitude, 
        height=np.max(current_magnitude)*0.05,  # 5% of max
        distance=len(current_magnitude)//50,    # Minimum separation
        prominence=np.max(current_magnitude)*0.03
    )
    
    if len(current_peaks_idx) > 0:
        current_peak_data = []
        for i, peak_idx in enumerate(current_peaks_idx[:8]):  # Top 8 peaks
            freq_val = current_positive_freqs[peak_idx]
            mag_val = current_magnitude[peak_idx]
            if freq_val > 0.01:  # Skip very low frequencies
                current_peak_data.append({
                    'Peak': i+1,
                    'Frequency (Hz)': f"{freq_val:.4f}",
                    'Magnitude': f"{mag_val:.4f}",
                    'Rel. Amplitude': f"{mag_val/np.max(current_magnitude):.3f}"
                })
        
        if current_peak_data:
            df_current_peaks = pd.DataFrame(current_peak_data)
            st.dataframe(df_current_peaks, hide_index=True, use_container_width=True)
        else:
            st.info(f"No significant peaks detected for Q{probe_qubit} above threshold")
    else:
        st.info(f"No peaks detected in Q{probe_qubit} FFT analysis")
        

        
    # Comparison view
    st.markdown("---")
    st.subheader("📈 Qubit Comparison View")
    
    comparison_mode = st.radio(
        "Select comparison mode:",
        ["Overlay All", "Individual Subplots", "Statistical Summary"],
        horizontal=True
    )
    
    if comparison_mode == "Overlay All":
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(7):
            if i == probe_qubit:
                # Highlight probe qubit
                ax.plot(times, probs[:, i], color=colors[i], linewidth=2.5, 
                       label=f'Q{i} (PROBE)', alpha=1.0, zorder=10)
            else:
                ax.plot(times, probs[:, i], color=colors[i], linewidth=1.2, 
                       label=f'Q{i}', alpha=0.7, zorder=5)
        ax.set_xlabel('Time')
        ax.set_ylabel('P(1)')
        ax.set_title(f'All Top Qubits - Overlay Comparison (Q{probe_qubit} highlighted)')
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=7, loc='upper right')
        st.pyplot(fig)
        
    elif comparison_mode == "Individual Subplots":
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(7):
            linewidth = 2.5 if i == probe_qubit else 1.5
            alpha = 1.0 if i == probe_qubit else 0.8
            title_suffix = " (PROBE)" if i == probe_qubit else ""
            
            axes[i].plot(times, probs[:, i], color=colors[i], 
                       linewidth=linewidth, alpha=alpha)
            axes[i].set_title(f'Q{i}{title_suffix}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('P(1)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-0.05, 1.05)
            
            # Add border highlight for probe qubit
            if i == probe_qubit:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
        
        # Hide the last subplot
        axes[7].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif comparison_mode == "Statistical Summary":
        # Create comprehensive statistics
        summary_data = []
        for i in range(7):
            signal = probs[:, i]
            qubit_label = f'Q{i} (PROBE)' if i == probe_qubit else f'Q{i}'
            summary_data.append({
                'Qubit': qubit_label,
                'Mean': f'{np.mean(signal):.4f}',
                'Std Dev': f'{np.std(signal):.4f}',
                'Min': f'{np.min(signal):.4f}',
                'Max': f'{np.max(signal):.4f}',
                'Range': f'{np.max(signal) - np.min(signal):.4f}',
                'Variance': f'{np.var(signal):.4f}'
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, hide_index=True, use_container_width=True)
        
        # Add correlation matrix
        st.markdown("**Qubit-Qubit Correlation Matrix:**")
        corr_matrix = np.corrcoef([probs[:, i] for i in range(7)])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels([f'Q{i}' for i in range(7)])
        ax.set_yticklabels([f'Q{i}' for i in range(7)])
        
        # Add correlation values as text
        for i in range(7):
            for j in range(7):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im)
        ax.set_title('Qubit Occupation Probability Correlations')
        st.pyplot(fig)

else:
    st.info("👆 Run a simulation first to see the interactive multi-qubit analysis!")

# PDF Report Generation Section - ADD THIS HERE
if 'simulation_results' in st.session_state:
    st.markdown("---")
    st.header("📄 Generate PDF Report")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("Generate a comprehensive PDF report with all analysis, graphs, and insights from your simulation.")
        
        report_filename = st.text_input(
            "Report filename:", 
            value=f"quantum_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    
    with col2:
        if st.button("📥 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating comprehensive PDF report..."):
                # Prepare parameters
                parameters = {
                    'k_pattern': k_pattern,
                    'j_couplings': j_couplings,
                    'j_mode': j_mode,
                    'dt': dt,
                    't_max': t_max,
                    'probe_qubit': probe_qubit,
                    'initial_amplitudes': {
                        'a': a_complex,
                        'b': b_complex,
                        'c': c_complex
                    }
                }
                
                # Generate report
                try:
                    from quantum_report_generator import generate_quantum_report
                    
                    report_path = generate_quantum_report(
                        st.session_state.simulation_results,
                        parameters,
                        report_filename
                    )
                    
                    st.success(f"✅ Report generated successfully: {report_path}")
                    
                    # Offer download
                    with open(report_path, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=PDFbyte,
                        file_name=report_filename,
                        mime='application/pdf'
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# Footer
st.markdown("---")
st.info("🔧 **Dynamic Configuration**: Select any 3 top qubits (0-6) to connect to bottom qubits (7,8,9). "
        "The initial quantum state automatically adapts to always include qubit 0 plus one of your selected qubits. "
        "This flexible design allows you to explore different coupling topologies while maintaining the same "
        "analytical framework. PST and custom J-coupling options are fully supported.")