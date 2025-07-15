"""
6-Qubit K-Pattern Diagnostic App - Fixed Version
Simplified version focused on understanding k-pattern effects on quantum dynamics
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
from scipy.signal import find_peaks
import copy
import time

# Page configuration
st.set_page_config(
    page_title="6-Qubit K-Pattern Diagnostic",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .pattern-easy { background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50; }
    .pattern-hard { background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #f44336; }
    .physics-insight { background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; }
    .debug-info { background-color: #fff3e0; padding: 10px; border-radius: 5px; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# QUANTUM SIMULATION FUNCTIONS
# ==========================================

def get_full_basis_6qubits():
    """Generate all 64 basis states for 6 qubits"""
    basis_states = list(product([0, 1], repeat=6))
    state_to_idx = {s: i for i, s in enumerate(basis_states)}
    return basis_states, state_to_idx

def commutator(H, rho):
    """Calculate commutator [H, rho]"""
    return -1j * (H @ rho - rho @ H)

def rk4_step(H, rho, dt):
    """Runge-Kutta 4th order integration step"""
    k1 = dt * commutator(H, rho)
    k2 = dt * commutator(H, rho + 0.5 * k1)
    k3 = dt * commutator(H, rho + 0.5 * k2)
    k4 = dt * commutator(H, rho + k3)
    return rho + (k1 + 2*k2 + 2*k3 + k4) / 6

def initialize_superposition(basis_states, state_to_idx, superposition_dict):
    """Initialize quantum state as superposition"""
    psi = np.zeros(len(basis_states), complex)
    for state, amp in superposition_dict.items():
        if state in state_to_idx:
            psi[state_to_idx[state]] = amp
    
    # Normalize
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi /= norm
    
    rho = np.outer(psi, psi.conj())
    return psi, rho

def calculate_zz_energy(state, k_pattern):
    """Calculate diagonal energy from ZZ interactions"""
    energy = 0.0
    for (i, j), k_val in k_pattern.items():
        # ZZ interaction: +k if spins aligned, -k if anti-aligned
        if state[i] == state[j]:
            energy += k_val
        else:
            energy -= k_val
    return energy

def generate_hamiltonian_6qubits(basis_states, state_to_idx, k_pattern, j_coupling, debug=False):
    """
    Generate Hamiltonian with specific k_pattern
    NO CACHING - fresh generation each time
    """
    n = len(basis_states)
    H = np.zeros((n, n), complex)
    
    if debug:
        st.markdown("**Debug Info:** Building Hamiltonian")
        st.write("K-pattern:", k_pattern)
    
    # Diagonal elements (ZZ interactions)
    for idx, state in enumerate(basis_states):
        H[idx, idx] = calculate_zz_energy(state, k_pattern)
    
    # Off-diagonal elements (XX+YY interactions) - horizontal chain only
    j_pairs = [(0, 2), (2, 4)]  # Q0-Q2-Q4 chain
    
    for (i, j) in j_pairs:
        for idx, state in enumerate(basis_states):
            if state[i] != state[j]:  # Can flip
                # Create flipped state
                flipped = list(state)
                flipped[i], flipped[j] = state[j], state[i]
                idx2 = state_to_idx[tuple(flipped)]
                
                # Add J-coupling
                H[idx, idx2] += j_coupling
                H[idx2, idx] += j_coupling
    
    # Diagonalize
    evals, evecs = np.linalg.eigh(H)
    
    if debug:
        st.write(f"Eigenvalue range: [{evals[0]:.3f}, {evals[-1]:.3f}]")
    
    return H, evals, evecs

def analyze_fft_peaks(probs, times, qubit_idx):
    """Enhanced FFT analysis to find frequency peaks with better resolution"""
    # Check if there's any oscillation
    if np.std(probs[:, qubit_idx]) < 0.001:
        return [], (np.array([0]), np.array([0]))
    
    dt = times[1] - times[0]
    
    # Remove DC component
    sig = probs[:, qubit_idx] - np.mean(probs[:, qubit_idx])
    
    # Apply Hanning window
    win = np.hanning(len(sig))
    sig_windowed = sig * win
    
    # FFT with zero padding for better frequency resolution
    n_fft = 2 * len(sig)  # Zero pad for better resolution
    fft = np.fft.fft(sig_windowed, n_fft)
    freq = np.fft.fftfreq(n_fft, dt)
    
    # Keep only positive frequencies
    mask = freq >= 0
    pos_freq = freq[mask]
    pos_mag = np.abs(fft[mask])
    
    # Find peaks with more sensitive parameters
    threshold = 0.05 * np.max(pos_mag) if np.max(pos_mag) > 0 else 0
    peaks, properties = find_peaks(pos_mag, height=threshold, distance=3, prominence=threshold*0.5)
    
    # Skip DC component and very low frequencies
    peaks = peaks[pos_freq[peaks] > 0.05]
    
    # Sort peaks by amplitude (descending)
    if len(peaks) > 0:
        peak_amplitudes = pos_mag[peaks]
        sorted_indices = np.argsort(peak_amplitudes)[::-1]
        peaks = peaks[sorted_indices]
    
    return peaks, (pos_freq, pos_mag)

def run_k_pattern_simulation(k_pattern, pattern_name, a_complex, b_complex, c_complex, 
                           j_coupling, dt=0.01, t_max=50, debug=False):
    """Run simulation for a specific k-pattern"""
    
    if debug:
        st.write(f"**Starting simulation for pattern: {pattern_name}**")
        st.write(f"K-values: {dict(k_pattern)}")
    
    # Get basis
    basis_states, state_to_idx = get_full_basis_6qubits()
    
    # Create projectors for probability measurements
    projectors = []
    for q in range(6):
        proj = np.zeros((64, 64))
        for idx, state in enumerate(basis_states):
            if state[q] == 1:
                proj[idx, idx] = 1.0
        projectors.append(proj)
    
    # Initial states (as specified in original)
    state_A = (1, 1, 0, 0, 0, 0)  # Q0=1, Q1=1
    state_B = (1, 0, 0, 1, 0, 0)  # Q0=1, Q3=1  
    state_C = (1, 0, 0, 0, 0, 1)  # Q0=1, Q5=1
    
    superposition_dict = {
        state_A: a_complex,
        state_B: b_complex,
        state_C: c_complex
    }
    
    # Initialize state
    psi0, rho0 = initialize_superposition(basis_states, state_to_idx, superposition_dict)
    
    # Generate Hamiltonian - FRESH COPY OF K_PATTERN
    k_pattern_copy = copy.deepcopy(k_pattern)
    H, evals, evecs = generate_hamiltonian_6qubits(
        basis_states, state_to_idx, k_pattern_copy, j_coupling, debug=debug
    )
    
    # Time evolution
    times = np.arange(0, t_max + dt, dt)
    n_steps = len(times)
    probs = np.zeros((n_steps, 6))
    
    # Initial probabilities
    for q in range(6):
        probs[0, q] = np.real(np.trace(rho0 @ projectors[q]))
    
    # Time evolution
    rho = rho0.copy()
    for i in range(1, n_steps):
        rho = rk4_step(H, rho, dt)
        for q in range(6):
            probs[i, q] = np.real(np.trace(rho @ projectors[q]))
    
    # Analyze FFT for monitoring qubits (Q0, Q2, Q4)
    fft_results = {}
    for q in [0, 2, 4]:
        peaks, fft_data = analyze_fft_peaks(probs, times, q)
        fft_results[q] = {
            'peaks': peaks, 
            'fft_data': fft_data,
            'n_peaks': len(peaks),
            'max_freq': fft_data[0][peaks[0]] if len(peaks) > 0 else 0
        }
    
    # Calculate some pattern characteristics
    total_k = sum(k_pattern.values())
    k_variance = np.var(list(k_pattern.values()))
    
    return {
        'pattern_name': pattern_name,
        'k_pattern': dict(k_pattern),
        'times': times,
        'probs': probs,
        'fft_results': fft_results,
        'eigenvalues': evals[:10],  # First 10 eigenvalues
        'total_k': total_k,
        'k_variance': k_variance
    }

# ==========================================
# K-PATTERN DEFINITIONS
# ==========================================

def get_diagnostic_k_patterns():
    """Define the k-patterns for diagnostic analysis - now with k=0 baseline"""
    patterns = {
        'k_zero_baseline': {(0,1): 0.0, (2,3): 0.0, (4,5): 0.0},  # NEW!
        'gradient_up': {(0,1): 0.5, (2,3): 1.25, (4,5): 2.0},
        'gradient_down': {(0,1): 2.0, (2,3): 1.25, (4,5): 0.5},
        'gradient_moderate': {(0,1): 0.7, (2,3): 1.0, (4,5): 1.9},  # NEW PATTERN!
        'uniform_1.5': {(0,1): 1.5, (2,3): 1.5, (4,5): 1.5},
        'well_shallow': {(0,1): 1.5, (2,3): 0.5, (4,5): 1.5},
        'barrier_low': {(0,1): 0.5, (2,3): 1.5, (4,5): 0.5}
    }
    
    # Verify patterns are different
    st.write("**K-Pattern Verification:**")
    for name, pattern in patterns.items():
        vals = list(pattern.values())
        st.write(f"- {name}: {vals} (sum={sum(vals):.1f}, var={np.var(vals):.2f})")
    
    return patterns

def draw_6qubit_topology():
    """Draw the 6-qubit topology"""
    fig = go.Figure()
    
    # Qubit positions (2x3 grid)
    qubit_x = [0, 0, 2, 2, 4, 4]
    qubit_y = [1, 0, 1, 0, 1, 0]
    qubit_labels = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    qubit_colors = ['lightblue', 'lightpink', 'lightblue', 'lightpink', 'lightblue', 'lightpink']
    
    # Add qubits
    fig.add_trace(go.Scatter(
        x=qubit_x, y=qubit_y,
        mode='markers+text',
        marker=dict(size=60, color=qubit_colors, line=dict(width=2, color='darkgray')),
        text=qubit_labels,
        textposition="middle center",
        textfont=dict(size=14, color='black'),
        showlegend=False
    ))
    
    # J-couplings (horizontal)
    j_connections = [(0, 2), (2, 4)]
    for i, j in j_connections:
        fig.add_trace(go.Scatter(
            x=[qubit_x[i], qubit_x[j]], 
            y=[qubit_y[i], qubit_y[j]],
            mode='lines',
            line=dict(width=3, color='darkgreen'),
            showlegend=False
        ))
    
    # k-couplings (vertical)
    k_connections = [(0, 1), (2, 3), (4, 5)]
    for i, j in k_connections:
        fig.add_trace(go.Scatter(
            x=[qubit_x[i], qubit_x[j]], 
            y=[qubit_y[i], qubit_y[j]],
            mode='lines',
            line=dict(width=4, color='red'),
            showlegend=False
        ))
    
    # Legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', 
                           line=dict(width=4, color='red'), name='k-coupling (ZZ)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', 
                           line=dict(width=3, color='darkgreen'), name='J-coupling (XX+YY)'))
    
    fig.update_layout(
        title="6-Qubit System: Monitor Top Chain Q0→Q2→Q4",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
        height=300,
        plot_bgcolor='white'
    )
    
    return fig

# ==========================================
# MAIN APP
# ==========================================

st.title("🔬 6-Qubit K-Pattern Diagnostic Analysis")
st.markdown("**Understanding Why Some K-Patterns Are Harder for ML**")

# Show topology
st.plotly_chart(draw_6qubit_topology(), use_container_width=True)

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Input parameters
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parameters")
    
    # Quantum amplitudes
    st.markdown("**Complex Amplitudes:**")
    a_real = st.number_input("a_real", value=0.148, format="%.3f")
    a_imag = st.number_input("a_imag", value=2.026, format="%.3f")
    b_real = st.number_input("b_real", value=-2.004, format="%.3f")
    b_imag = st.number_input("b_imag", value=0.294, format="%.3f")
    c_real = st.number_input("c_real", value=1.573, format="%.3f")
    c_imag = st.number_input("c_imag", value=-2.555, format="%.3f")
    
    st.markdown("**Simulation:**")
    j_coupling = st.slider("J-coupling", 0.0, 2.0, 0.5, 0.1)
    t_max = st.slider("Max time", 10.0, 100.0, 50.0, 5.0)
    
    # Initialize dt with default value
    dt = 0.01
    
    with st.expander("Advanced FFT Settings"):
        dt = st.number_input("Time step (dt)", value=0.01, min_value=0.001, max_value=0.1, step=0.001, format="%.3f")
        st.info("Smaller dt gives better time resolution but increases computation time")

with col2:
    st.subheader("K-Patterns to Test")
    
    # Get and display patterns
    k_patterns = get_diagnostic_k_patterns()
    
    col_easy, col_hard = st.columns(2)
    with col_easy:
        st.markdown('<div class="pattern-easy">', unsafe_allow_html=True)
        st.markdown("**🟢 Baseline & Easy Patterns**")
        st.markdown("- **k_zero_baseline**: 0.0 → 0.0 → 0.0")  # NEW!
        st.markdown("- **gradient_up**: 0.5 → 1.25 → 2.0")
        st.markdown("- **gradient_down**: 2.0 → 1.25 → 0.5")
        st.markdown("- **gradient_moderate**: 0.7 → 1.0 → 1.9")  # NEW PATTERN!
        st.markdown("*No coupling / Smooth gradients*")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_hard:
        st.markdown('<div class="pattern-hard">', unsafe_allow_html=True)
        st.markdown("**🔴 Hard Patterns (Expected poor ML)**")
        st.markdown("- **uniform_1.5**: 1.5 → 1.5 → 1.5")
        st.markdown("- **well_shallow**: 1.5 → 0.5 → 1.5")
        st.markdown("- **barrier_low**: 0.5 → 1.5 → 0.5")
        st.markdown("*Uniform or non-monotonic*")
        st.markdown("</div>", unsafe_allow_html=True)

# Run analysis
if st.button("🚀 Run K-Pattern Analysis", use_container_width=True):
    
    # Complex amplitudes
    a_complex = a_real + 1j * a_imag
    b_complex = b_real + 1j * b_imag
    c_complex = c_real + 1j * c_imag
    
    # Store results
    results = {}
    
    with st.spinner("Running simulations for all k-patterns..."):
        progress_bar = st.progress(0)
        
        # Run each pattern
        for i, (name, k_pattern) in enumerate(k_patterns.items()):
            progress_bar.progress((i + 1) / len(k_patterns))
            
            # Make a fresh copy of k_pattern
            k_pattern_copy = copy.deepcopy(k_pattern)
            
            if debug_mode:
                st.write(f"---\n**Pattern {i+1}/{len(k_patterns)}: {name}**")
            
            # Run simulation
            result = run_k_pattern_simulation(
                k_pattern_copy, name, a_complex, b_complex, c_complex, 
                j_coupling, dt=dt, t_max=t_max, debug=debug_mode
            )
            
            results[name] = result
    
    st.success("✅ All simulations complete!")
    
    # ==========================================
    # VERIFICATION SECTION
    # ==========================================
    
    st.markdown("---")
    st.header("🔍 Pattern Verification")
    
    # Show that patterns are indeed different
    verification_data = []
    for name, result in results.items():
        verification_data.append({
            'Pattern': name,
            'k(0,1)': result['k_pattern'][(0,1)],
            'k(2,3)': result['k_pattern'][(2,3)],
            'k(4,5)': result['k_pattern'][(4,5)],
            'Total K': result['total_k'],
            'K Variance': f"{result['k_variance']:.3f}",
            'First Eigenvalue': f"{result['eigenvalues'][0]:.3f}",
            'Q4 Peaks': result['fft_results'][4]['n_peaks']
        })
    
    df_verify = pd.DataFrame(verification_data)
    st.dataframe(df_verify, use_container_width=True)
    
    # ==========================================
    # ANALYSIS AND VISUALIZATION
    # ==========================================
    
    st.markdown("---")
    st.header("📊 Comparative Analysis Results")
    
    # 1. Occupation Probability Dynamics
    st.subheader("1. Occupation Probability Dynamics (Top Qubits)")
    
    # FIXED: Changed from 3x5 to 3x7 to accommodate all 7 patterns
    fig = make_subplots(
        rows=3, cols=7,
        subplot_titles=list(k_patterns.keys()),
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.05
    )
    
    colors = ['blue', 'green', 'red']
    qubit_names = ['Q0', 'Q2', 'Q4']
    
    for col, (name, result) in enumerate(results.items(), 1):
        times = result['times']
        probs = result['probs']
        
        for row, q in enumerate([0, 2, 4], 1):
            fig.add_trace(
                go.Scatter(x=times, y=probs[:, q], 
                          line=dict(color=colors[row-1], width=2),
                          name=f"{qubit_names[row-1]}" if col == 1 else "",
                          showlegend=(col == 1)),
                row=row, col=col
            )
    
    fig.update_layout(height=600, title="Probability Dynamics Across K-Patterns")
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. FFT Spectra Comparison
    st.subheader("2. FFT Spectra Comparison")
    
    # Add visualization options
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        normalize_fft = st.checkbox("Normalize FFT amplitudes", value=False)
    with col_opt2:
        show_peak_details = st.checkbox("Show peak details", value=True)
    with col_opt3:
        selected_pattern = st.selectbox("Focus on pattern:", ["All"] + list(k_patterns.keys()))
    
    # FIXED: Changed from 3x5 to 3x7 to accommodate all 7 patterns
    fig_fft = make_subplots(
        rows=3, cols=7,
        subplot_titles=list(k_patterns.keys()),
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.05
    )
    
    for col, (name, result) in enumerate(results.items(), 1):
        for row, q in enumerate([0, 2, 4], 1):
            freq, mag = result['fft_results'][q]['fft_data']
            peaks = result['fft_results'][q]['peaks']
            
            # Normalize if requested
            if normalize_fft and np.max(mag) > 0:
                mag = mag / np.max(mag)
            
            # Plot FFT
            fig_fft.add_trace(
                go.Scatter(x=freq, y=mag,
                          line=dict(color=colors[row-1], width=1.5),
                          name=f"{qubit_names[row-1]}" if col == 1 else "",
                          showlegend=(col == 1)),
                row=row, col=col
            )
            
            # Mark peaks
            if len(peaks) > 0:
                fig_fft.add_trace(
                    go.Scatter(x=freq[peaks], y=mag[peaks],
                              mode='markers',
                              marker=dict(color='red', size=6),
                              showlegend=False),
                    row=row, col=col
                )
    
    fig_fft.update_layout(height=600, title="FFT Spectra Across K-Patterns")
    fig_fft.update_xaxes(range=[0, 2])
    st.plotly_chart(fig_fft, use_container_width=True)
    
    # Show detailed peak information
    if show_peak_details:
        st.subheader("2.1 FFT Peak Details")
        
        # Create detailed peak table
        peak_data = []
        for name, result in results.items():
            if selected_pattern == "All" or selected_pattern == name:
                for q in [0, 2, 4]:
                    freq, mag = result['fft_results'][q]['fft_data']
                    peaks = result['fft_results'][q]['peaks']
                    
                    if len(peaks) > 0:
                        # Get top 5 peaks sorted by amplitude
                        peak_mags = mag[peaks]
                        sorted_indices = np.argsort(peak_mags)[::-1][:5]
                        
                        for i, idx in enumerate(sorted_indices):
                            peak_idx = peaks[idx]
                            peak_data.append({
                                'Pattern': name,
                                'Qubit': f'Q{q}',
                                'Peak': i + 1,
                                'Frequency (Hz)': f"{freq[peak_idx]:.3f}",
                                'Amplitude': f"{mag[peak_idx]:.3f}",
                                'Normalized Amp': f"{mag[peak_idx]/np.max(mag) if np.max(mag) > 0 else 0:.3f}"
                            })
        
        df_peaks = pd.DataFrame(peak_data)
        
        if selected_pattern != "All":
            st.markdown(f"**Peaks for {selected_pattern}:**")
            pattern_df = df_peaks[df_peaks['Pattern'] == selected_pattern]
            st.dataframe(pattern_df, use_container_width=True)
            
            # Special focus plot for selected pattern
            st.subheader(f"2.2 Detailed FFT for {selected_pattern}")
            
            result = results[selected_pattern]
            fig_detail = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Q0', 'Q2', 'Q4'],
                shared_yaxes=False
            )
            
            for col, q in enumerate([0, 2, 4], 1):
                freq, mag = result['fft_results'][q]['fft_data']
                peaks = result['fft_results'][q]['peaks']
                
                # Plot FFT
                fig_detail.add_trace(
                    go.Scatter(x=freq, y=mag,
                              line=dict(color=colors[col-1], width=2),
                              name=f'Q{q}'),
                    row=1, col=col
                )
                
                # Mark and annotate peaks
                if len(peaks) > 0:
                    fig_detail.add_trace(
                        go.Scatter(x=freq[peaks], y=mag[peaks],
                                  mode='markers+text',
                                  marker=dict(color='red', size=8),
                                  text=[f"{freq[p]:.3f}Hz<br>{mag[p]:.2f}" for p in peaks],
                                  textposition="top center",
                                  showlegend=False),
                        row=1, col=col
                    )
            
            fig_detail.update_layout(height=400, title=f"Detailed FFT Analysis for {selected_pattern}")
            fig_detail.update_xaxes(range=[0, 2])
            st.plotly_chart(fig_detail, use_container_width=True)
            
        else:
            st.dataframe(df_peaks, use_container_width=True)
    
    # 3. Signal Strength Analysis
    st.subheader("3. Signal Strength Analysis")
    
    # Calculate signal metrics
    signal_data = []
    for name, result in results.items():
        probs = result['probs']
        for q in [0, 2, 4]:
            oscillation = np.max(probs[:, q]) - np.min(probs[:, q])
            avg_prob = np.mean(probs[:, q])
            signal_data.append({
                'Pattern': name,
                'Qubit': f'Q{q}',
                'Oscillation': oscillation,
                'Avg_Probability': avg_prob,
                'Signal_Strength': oscillation * avg_prob
            })
    
    df_signals = pd.DataFrame(signal_data)
    
    # Create heatmap for oscillation amplitude
    pivot_osc = df_signals.pivot(index='Qubit', columns='Pattern', values='Oscillation')
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_osc.values,
        x=pivot_osc.columns,
        y=pivot_osc.index,
        colorscale='Viridis',
        text=np.round(pivot_osc.values, 3),
        texttemplate="%{text}",
        textfont={"size":10}
    ))
    
    fig_heat.update_layout(
        title="Oscillation Amplitude Heatmap",
        height=300
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # 4. Eigenvalue Analysis
    st.subheader("4. Eigenvalue Spectrum Comparison")
    
    fig_eigen = go.Figure()
    
    for name, result in results.items():
        fig_eigen.add_trace(go.Scatter(
            x=list(range(len(result['eigenvalues']))),
            y=result['eigenvalues'],
            mode='lines+markers',
            name=name,
            line=dict(width=2)
        ))
    
    fig_eigen.update_layout(
        title="First 10 Eigenvalues for Each Pattern",
        xaxis_title="Eigenvalue Index",
        yaxis_title="Energy",
        height=400
    )
    
    st.plotly_chart(fig_eigen, use_container_width=True)
    
    # 5. Physics Analysis Report
    st.subheader("5. Physics Analysis Report")
    
    # Special comparison section for gradient_moderate
    if 'gradient_moderate' in results:
        st.markdown("### 🔍 Verification for gradient_moderate (0.7, 1.0, 1.9)")
        
        result_gm = results['gradient_moderate']
        st.markdown("**Your expected peaks vs. simulation results:**")
        
        expected_peaks = [
            ("Peak 1", 0.340, 9.416),
            ("Peak 2", 0.340, 9.416),
            ("Peak 3", 0.748, 4.019),
            ("Peak 4", 0.557, 3.311),
            ("Peak 5", 1.084, 2.808)
        ]
        
        # Get actual peaks for Q4 (or whichever qubit you measured)
        for q in [4]:  # Focus on Q4
            freq, mag = result_gm['fft_results'][q]['fft_data']
            peaks = result_gm['fft_results'][q]['peaks']
            
            st.markdown(f"**For Qubit Q{q}:**")
            
            if len(peaks) > 0:
                # Create comparison table
                comparison_data = []
                
                # Show top 5 peaks
                for i in range(min(5, len(peaks))):
                    peak_idx = peaks[i]
                    comparison_data.append({
                        'Peak': f'Peak {i+1}',
                        'Expected Freq (Hz)': expected_peaks[i][1] if i < len(expected_peaks) else '-',
                        'Expected Amp': expected_peaks[i][2] if i < len(expected_peaks) else '-',
                        'Simulated Freq (Hz)': f"{freq[peak_idx]:.3f}",
                        'Simulated Amp': f"{mag[peak_idx]:.3f}",
                        'Freq Diff': f"{abs(freq[peak_idx] - expected_peaks[i][1]):.3f}" if i < len(expected_peaks) else '-'
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                # Note about potential differences
                st.info("""
                **Note on differences:** Small variations in peak frequencies and amplitudes can occur due to:
                - Simulation time step (dt) and total time
                - FFT resolution and windowing
                - Numerical precision in the evolution
                - Initial state normalization
                """)
    
    st.markdown('<div class="physics-insight">', unsafe_allow_html=True)

    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Store results for further analysis
    st.session_state.diagnostic_results = results
    
else:
    st.info("👆 Click 'Run K-Pattern Analysis' to start the comparative study!")
    
# ==========================================
# STANDALONE MATHEMATICAL HAMILTONIAN ANALYSIS SECTION
# ==========================================
# 
# INSTALLATION INSTRUCTIONS:
# 1. Add this code AFTER the main "if st.button" block (after the "else:" statement)
# 2. Place it BEFORE the final "st.markdown("---")" line
# 3. This will make it always available, regardless of whether the button was clicked
#
# ==========================================

# Mathematical Hamiltonian Analysis Section (Always Available)
st.markdown("---")
st.header("🧮 Mathematical Hamiltonian Analysis")
st.markdown("**Understanding Why Gradient K-Couplings Are More Predictable**")

st.markdown("""
<div class="physics-insight">
<h4>🔬 Mathematical Foundation</h4>
<p>The 6-qubit Hamiltonian has the form:</p>
<p><strong>H = H_ZZ + H_XX+YY</strong></p>
<p>where:</p>
<ul>
    <li><strong>H_ZZ = k₀₁Z₀Z₁ + k₂₃Z₂Z₃ + k₄₅Z₄Z₅</strong> (diagonal terms)</li>
    <li><strong>H_XX+YY = J(X₀X₂ + Y₀Y₂ + X₂X₄ + Y₂Y₄)</strong> (off-diagonal terms)</li>
</ul>
<p>The spectral variance comes from how quantum probability flows through different K-coupling energy landscapes.</p>
</div>
""", unsafe_allow_html=True)

# Get k_patterns for analysis
k_patterns = get_diagnostic_k_patterns()

# Pattern selection for comparison
st.markdown("#### 🔍 Pattern Comparison Analysis")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    pattern1 = st.selectbox("Select Gradient Pattern:", 
                           [name for name in k_patterns.keys() if 'gradient' in name],
                           index=0,
                           key="math_pattern1")

with col_comp2:
    pattern2 = st.selectbox("Select Symmetric Pattern:", 
                           [name for name in k_patterns.keys() if name in ['uniform_1.5', 'well_shallow', 'barrier_low']],
                           index=2,
                           key="math_pattern2")

# Get pattern data
pattern1_data = k_patterns[pattern1]
pattern2_data = k_patterns[pattern2]

# Display pattern characteristics
st.markdown("#### 📊 Pattern Characteristics")

# Calculate pattern metrics
def calculate_pattern_metrics(pattern_data):
    k_vals = list(pattern_data.values())
    return {
        'k_01': k_vals[0],
        'k_23': k_vals[1], 
        'k_45': k_vals[2],
        'k_gradient': k_vals[2] - k_vals[0],
        'k_variance': np.var(k_vals),
        'k_symmetry': abs(k_vals[0] - k_vals[2]),
        'k_total': sum(k_vals),
        'k_range': max(k_vals) - min(k_vals)
    }

metrics1 = calculate_pattern_metrics(pattern1_data)
metrics2 = calculate_pattern_metrics(pattern2_data)

# Comparison table - ensure all values are strings for consistent display
comp_data = {
    'Metric': ['K-coupling [0,1]', 'K-coupling [2,3]', 'K-coupling [4,5]', 
              'K Gradient (k₄₅-k₀₁)', 'K Variance', 'K Symmetry |k₀₁-k₄₅|', 'K Range'],
    pattern1: [f"{metrics1['k_01']:.2f}", f"{metrics1['k_23']:.2f}", f"{metrics1['k_45']:.2f}", 
              f"{metrics1['k_gradient']:.2f}", f"{metrics1['k_variance']:.3f}", 
              f"{metrics1['k_symmetry']:.2f}", f"{metrics1['k_range']:.3f}"],
    pattern2: [f"{metrics2['k_01']:.2f}", f"{metrics2['k_23']:.2f}", f"{metrics2['k_45']:.2f}", 
              f"{metrics2['k_gradient']:.2f}", f"{metrics2['k_variance']:.3f}", 
              f"{metrics2['k_symmetry']:.2f}", f"{metrics2['k_range']:.3f}"]
}

df_comp = pd.DataFrame(comp_data)

# Display with error handling
try:
    st.dataframe(df_comp, use_container_width=True)
except Exception as e:
    st.warning(f"Display issue with comparison table: {e}")
    # Fallback to simple table display
    for metric, val1, val2 in zip(comp_data['Metric'], comp_data[pattern1], comp_data[pattern2]):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(metric)
        with col2:
            st.write(f"{pattern1}: {val1}")
        with col3:
            st.write(f"{pattern2}: {val2}")

# ==========================================
# FULL HAMILTONIAN MATRIX VISUALIZATION
# ==========================================

st.markdown("#### 🔲 Full Hamiltonian Matrix Visualization")

st.markdown("""
<div class="physics-insight">
<h4>🎯 Complete Hamiltonian Structure</h4>
<p>The full 64×64 Hamiltonian matrix H = H_ZZ + H_XX+YY shows the complete quantum mechanical structure:</p>
<ul>
    <li><strong>Diagonal elements:</strong> ZZ coupling energies</li>
    <li><strong>Off-diagonal elements:</strong> XX+YY transition amplitudes</li>
    <li><strong>Sparsity pattern:</strong> Reveals which quantum states are connected</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Calculate full Hamiltonian matrices for both patterns
def calculate_full_hamiltonian(k_01, k_23, k_45, j_coupling=0.5):
    """Calculate the full 6-qubit Hamiltonian matrix"""
    from itertools import product
    
    # Get 6-qubit basis
    basis_states = list(product([0, 1], repeat=6))
    state_to_idx = {s: i for i, s in enumerate(basis_states)}
    n = len(basis_states)
    
    # Initialize Hamiltonian
    H = np.zeros((n, n), complex)
    
    # K-pattern for ZZ terms
    k_pattern = {(0,1): k_01, (2,3): k_23, (4,5): k_45}
    
    # Diagonal elements (ZZ interactions)
    for idx, state in enumerate(basis_states):
        energy = 0.0
        for (i, j), k_val in k_pattern.items():
            if state[i] == state[j]:
                energy += k_val
            else:
                energy -= k_val
        H[idx, idx] = energy
    
    # Off-diagonal elements (XX+YY interactions)
    j_pairs = [(0, 2), (2, 4)]  # Q0-Q2-Q4 chain
    
    for (i, j) in j_pairs:
        for idx, state in enumerate(basis_states):
            if state[i] != state[j]:  # Can flip
                # Create flipped state
                flipped = list(state)
                flipped[i], flipped[j] = state[j], state[i]
                idx2 = state_to_idx[tuple(flipped)]
                
                # Add J-coupling
                H[idx, idx2] += j_coupling
                H[idx2, idx] += j_coupling
    
    return H, basis_states

# Calculate Hamiltonians for both patterns
try:
    H1, basis_states = calculate_full_hamiltonian(metrics1['k_01'], metrics1['k_23'], metrics1['k_45'])
    H2, _ = calculate_full_hamiltonian(metrics2['k_01'], metrics2['k_23'], metrics2['k_45'])
    
    # Display matrix properties
    st.markdown("##### 📊 Matrix Properties")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.markdown(f"**{pattern1} Hamiltonian:**")
        st.write(f"• Matrix size: {H1.shape[0]}×{H1.shape[1]}")
        st.write(f"• Non-zero elements: {np.count_nonzero(H1)}")
        st.write(f"• Sparsity: {(1 - np.count_nonzero(H1)/H1.size)*100:.1f}%")
        st.write(f"• Diagonal range: [{np.min(np.diag(H1.real)):.2f}, {np.max(np.diag(H1.real)):.2f}]")
        st.write(f"• Off-diagonal max: {np.max(np.abs(H1 - np.diag(np.diag(H1)))):.2f}")
        
    with col_h2:
        st.markdown(f"**{pattern2} Hamiltonian:**")
        st.write(f"• Matrix size: {H2.shape[0]}×{H2.shape[1]}")
        st.write(f"• Non-zero elements: {np.count_nonzero(H2)}")
        st.write(f"• Sparsity: {(1 - np.count_nonzero(H2)/H2.size)*100:.1f}%")
        st.write(f"• Diagonal range: [{np.min(np.diag(H2.real)):.2f}, {np.max(np.diag(H2.real)):.2f}]")
        st.write(f"• Off-diagonal max: {np.max(np.abs(H2 - np.diag(np.diag(H2)))):.2f}")
    
    # Matrix visualization options
    st.markdown("##### 🎨 Visualization Options")
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        matrix_view = st.selectbox("Matrix view:", 
                                  ["Real part", "Imaginary part", "Absolute value", "Diagonal only"],
                                  index=0)
    
    with col_opt2:
        color_scale = st.selectbox("Color scale:", 
                                  ["RdBu", "Viridis", "Plasma", "Blues", "RdYlBu"],
                                  index=0)
    
    with col_opt3:
        show_values = st.checkbox("Show matrix values", value=False)
    
    # Create matrix visualization
    def create_matrix_heatmap(H, title, matrix_view, color_scale, show_values):
        """Create heatmap of Hamiltonian matrix"""
        
        # Select matrix data based on view
        if matrix_view == "Real part":
            matrix_data = H.real
        elif matrix_view == "Imaginary part":
            matrix_data = H.imag
        elif matrix_view == "Absolute value":
            matrix_data = np.abs(H)
        elif matrix_view == "Diagonal only":
            matrix_data = np.diag(np.diag(H.real))
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            colorscale=color_scale,
            showscale=True,
            text=np.round(matrix_data, 3) if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 8} if show_values else None,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"{title} - {matrix_view}",
            xaxis_title="Basis State Index",
            yaxis_title="Basis State Index",
            height=500,
            width=500
        )
        
        return fig
    
    # Display matrix heatmaps
    st.markdown("##### 🌡️ Hamiltonian Matrix Heatmaps")
    
    col_matrix1, col_matrix2 = st.columns(2)
    
    with col_matrix1:
        fig1 = create_matrix_heatmap(H1, pattern1, matrix_view, color_scale, show_values)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_matrix2:
        fig2 = create_matrix_heatmap(H2, pattern2, matrix_view, color_scale, show_values)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Matrix difference analysis
    st.markdown("##### 🔍 Matrix Difference Analysis")
    
    # Calculate matrix difference
    H_diff = H1 - H2
    
    # Display difference statistics
    col_diff1, col_diff2, col_diff3 = st.columns(3)
    
    with col_diff1:
        st.metric("Max difference", f"{np.max(np.abs(H_diff)):.3f}")
        st.metric("RMS difference", f"{np.sqrt(np.mean(np.abs(H_diff)**2)):.3f}")
        
    with col_diff2:
        st.metric("Diagonal differences", f"{np.max(np.abs(np.diag(H_diff))):.3f}")
        st.metric("Off-diagonal differences", f"{np.max(np.abs(H_diff - np.diag(np.diag(H_diff)))):.3f}")
        
    with col_diff3:
        st.metric("Frobenius norm", f"{np.linalg.norm(H_diff, 'fro'):.3f}")
        st.metric("Spectral norm", f"{np.linalg.norm(H_diff, 2):.3f}")
    
    # Difference heatmap
    fig_diff = go.Figure(data=go.Heatmap(
        z=H_diff.real,
        colorscale="RdBu",
        showscale=True,
        zmid=0,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Difference: %{z:.3f}<extra></extra>"
    ))
    
    fig_diff.update_layout(
        title=f"Matrix Difference: {pattern1} - {pattern2}",
        xaxis_title="Basis State Index",
        yaxis_title="Basis State Index",
        height=500
    )
    
    st.plotly_chart(fig_diff, use_container_width=True)
    
    # Basis state information
    with st.expander("📋 Basis State Information"):
        st.markdown("**6-Qubit Basis States (first 20):**")
        
        # Display first 20 basis states
        basis_info = []
        for i in range(min(20, len(basis_states))):
            state = basis_states[i]
            diagonal_energy_1 = H1[i, i].real
            diagonal_energy_2 = H2[i, i].real
            
            basis_info.append({
                'Index': i,
                'State': '|' + ''.join(map(str, state)) + '⟩',
                'Binary': ''.join(map(str, state)),
                f'{pattern1} Energy': f"{diagonal_energy_1:.3f}",
                f'{pattern2} Energy': f"{diagonal_energy_2:.3f}",
                'Energy Diff': f"{diagonal_energy_1 - diagonal_energy_2:.3f}"
            })
        
        df_basis = pd.DataFrame(basis_info)
        st.dataframe(df_basis, use_container_width=True)
        
        if len(basis_states) > 20:
            st.info(f"Showing first 20 of {len(basis_states)} total basis states")
    
    # Matrix structure analysis
    st.markdown("##### 🏗️ Matrix Structure Analysis")
    
    # Matrix structure analysis
    st.markdown("##### 🏗️ Matrix Structure Analysis")
    
    # Add detailed calculation example
    st.markdown("##### 🔍 Step-by-Step Calculation Example")
    
    st.markdown("""
    <div class="physics-insight">
    <h4>📚 How to Calculate a Specific Matrix Element</h4>
    <p>Let's walk through the exact calculation of a diagonal element H(24,24) for the gradient_up pattern.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Let user select an index to analyze
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        example_index = st.number_input("Select matrix index to analyze:", 
                                       min_value=0, max_value=63, value=24, step=1)
    
    with col_calc2:
        analysis_pattern = st.selectbox("Pattern for calculation:", 
                                      [pattern1, pattern2], 
                                      index=0, key="calc_pattern")
    
    # Get pattern metrics for the selected pattern
    selected_metrics = metrics1 if analysis_pattern == pattern1 else metrics2
    
    # Calculate the specific matrix element
    def calculate_matrix_element_detailed(index, k_01, k_23, k_45):
        """Calculate matrix element with detailed explanation"""
        # Convert index to 6-bit binary (basis state)
        binary_str = format(index, '06b')
        basis_state = [int(b) for b in binary_str]
        
        # Calculate ZZ coupling energy
        s0, s1, s2, s3, s4, s5 = basis_state
        
        # Calculate each coupling term
        term_01 = k_01 if s0 == s1 else -k_01
        term_23 = k_23 if s2 == s3 else -k_23
        term_45 = k_45 if s4 == s5 else -k_45
        
        total_energy = term_01 + term_23 + term_45
        
        return {
            'index': index,
            'binary': binary_str,
            'basis_state': basis_state,
            'qubit_states': {'s0': s0, 's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5},
            'coupling_terms': {
                'term_01': term_01,
                'term_23': term_23,
                'term_45': term_45
            },
            'alignments': {
                'pair_01': 'aligned' if s0 == s1 else 'anti-aligned',
                'pair_23': 'aligned' if s2 == s3 else 'anti-aligned',
                'pair_45': 'aligned' if s4 == s5 else 'anti-aligned'
            },
            'total_energy': total_energy
        }
    
    # Perform the calculation
    calc_result = calculate_matrix_element_detailed(
        example_index, 
        selected_metrics['k_01'], 
        selected_metrics['k_23'], 
        selected_metrics['k_45']
    )
    
    # Display the step-by-step calculation
    st.markdown(f"**Step-by-Step Calculation for H({example_index},{example_index}) in {analysis_pattern} pattern:**")
    
    # Step 1: Index to basis state conversion
    st.markdown("**Step 1: Convert Index to Basis State**")
    st.code(f"""
Index {example_index} in binary (6 bits) = {calc_result['binary']}
Basis state: |{calc_result['binary']}⟩
    """)
    
    # Step 2: Qubit state analysis
    st.markdown("**Step 2: Analyze Qubit States**")
    qubit_display = " ".join([f"Q{i}={calc_result['qubit_states'][f's{i}']}" for i in range(6)])
    st.code(f"""
Qubit positions: Q0  Q1  Q2  Q3  Q4  Q5
State values:    {calc_result['basis_state'][0]}   {calc_result['basis_state'][1]}   {calc_result['basis_state'][2]}   {calc_result['basis_state'][3]}   {calc_result['basis_state'][4]}   {calc_result['basis_state'][5]}
Physical state:  {'↓' if calc_result['basis_state'][0] == 0 else '↑'}   {'↓' if calc_result['basis_state'][1] == 0 else '↑'}   {'↓' if calc_result['basis_state'][2] == 0 else '↑'}   {'↓' if calc_result['basis_state'][3] == 0 else '↑'}   {'↓' if calc_result['basis_state'][4] == 0 else '↑'}   {'↓' if calc_result['basis_state'][5] == 0 else '↑'}
    """)
    
    # Step 3: K-coupling values
    st.markdown("**Step 3: K-Coupling Values**")
    st.code(f"""
{analysis_pattern} pattern:
k_01 = {selected_metrics['k_01']:.2f}  (coupling between Q0,Q1)
k_23 = {selected_metrics['k_23']:.2f}  (coupling between Q2,Q3)
k_45 = {selected_metrics['k_45']:.2f}  (coupling between Q4,Q5)
    """)
    
    # Step 4: Calculate each coupling term
    st.markdown("**Step 4: Calculate ZZ Coupling Terms**")
    st.code(f"""
ZZ Energy Formula: E = k_01×(alignment_01) + k_23×(alignment_23) + k_45×(alignment_45)
Where alignment = +k if spins are same, -k if spins are different

Pair (0,1): Q0={calc_result['qubit_states']['s0']}, Q1={calc_result['qubit_states']['s1']} → {calc_result['alignments']['pair_01']} → {calc_result['coupling_terms']['term_01']:+.2f}
Pair (2,3): Q2={calc_result['qubit_states']['s2']}, Q3={calc_result['qubit_states']['s3']} → {calc_result['alignments']['pair_23']} → {calc_result['coupling_terms']['term_23']:+.2f}
Pair (4,5): Q4={calc_result['qubit_states']['s4']}, Q5={calc_result['qubit_states']['s5']} → {calc_result['alignments']['pair_45']} → {calc_result['coupling_terms']['term_45']:+.2f}
    """)
    
    # Step 5: Final calculation
    st.markdown("**Step 5: Final Energy Calculation**")
    st.code(f"""
Total ZZ Energy:
E_ZZ = {calc_result['coupling_terms']['term_01']:+.2f} + {calc_result['coupling_terms']['term_23']:+.2f} + {calc_result['coupling_terms']['term_45']:+.2f} = {calc_result['total_energy']:+.2f}

Therefore: H({example_index},{example_index}) = {calc_result['total_energy']:.2f}
    """)
    
    # Visual representation of the calculation
    st.markdown("**Visual Representation:**")
    
    # Create a visual diagram of the qubit layout with this specific state
    fig_qubit_state = go.Figure()
    
    # Qubit positions (2x3 grid)
    qubit_x = [0, 0, 2, 2, 4, 4]
    qubit_y = [1, 0, 1, 0, 1, 0]
    qubit_labels = [f'Q{i}' for i in range(6)]
    
    # Color qubits based on their states
    qubit_colors = ['lightblue' if calc_result['basis_state'][i] == 0 else 'lightcoral' for i in range(6)]
    qubit_text = [f'Q{i}={calc_result["basis_state"][i]}' for i in range(6)]
    
    # Add qubits
    fig_qubit_state.add_trace(go.Scatter(
        x=qubit_x, y=qubit_y,
        mode='markers+text',
        marker=dict(size=60, color=qubit_colors, line=dict(width=2, color='darkgray')),
        text=qubit_text,
        textposition="middle center",
        textfont=dict(size=12, color='black'),
        showlegend=False
    ))
    
    # Add coupling lines with energies
    coupling_info = [
        ((0, 1), calc_result['coupling_terms']['term_01'], 'red'),
        ((2, 3), calc_result['coupling_terms']['term_23'], 'red'),
        ((4, 5), calc_result['coupling_terms']['term_45'], 'red')
    ]
    
    for (i, j), energy, color in coupling_info:
        # Draw coupling line
        fig_qubit_state.add_trace(go.Scatter(
            x=[qubit_x[i], qubit_x[j]], 
            y=[qubit_y[i], qubit_y[j]],
            mode='lines',
            line=dict(width=4, color=color),
            showlegend=False
        ))
        
        # Add energy annotation
        mid_x = (qubit_x[i] + qubit_x[j]) / 2
        mid_y = (qubit_y[i] + qubit_y[j]) / 2
        fig_qubit_state.add_annotation(
            x=mid_x + 0.3, y=mid_y,
            text=f"{energy:+.2f}",
            showarrow=False,
            font=dict(size=12, color=color),
            bgcolor='white',
            bordercolor=color,
            borderwidth=1
        )
    
    fig_qubit_state.update_layout(
        title=f"Qubit State |{calc_result['binary']}⟩ with ZZ Coupling Energies",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
        height=300,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig_qubit_state, use_container_width=True)
    
    # Summary box
    st.markdown(f"""
    <div class="debug-info">
    <h4>📋 Calculation Summary</h4>
    <p><strong>Matrix Element:</strong> H({example_index},{example_index}) = {calc_result['total_energy']:.3f}</p>
    <p><strong>Basis State:</strong> |{calc_result['binary']}⟩</p>
    <p><strong>Physical Meaning:</strong> This is the energy eigenvalue of quantum state |{calc_result['binary']}⟩ 
    under the {analysis_pattern} K-coupling pattern.</p>
    <p><strong>Why This Matters:</strong> This diagonal element determines how this particular quantum state 
    contributes to the overall system dynamics and spectral properties.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Compare with other patterns
    if st.checkbox("Compare with other pattern"):
        other_pattern = pattern2 if analysis_pattern == pattern1 else pattern1
        other_metrics = metrics2 if analysis_pattern == pattern1 else metrics1
        
        other_result = calculate_matrix_element_detailed(
            example_index,
            other_metrics['k_01'],
            other_metrics['k_23'], 
            other_metrics['k_45']
        )
        
        st.markdown("**Pattern Comparison:**")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.metric(f"{analysis_pattern} H({example_index},{example_index})", 
                     f"{calc_result['total_energy']:.3f}")
        
        with comp_col2:
            st.metric(f"{other_pattern} H({example_index},{example_index})", 
                     f"{other_result['total_energy']:.3f}",
                     delta=f"{other_result['total_energy'] - calc_result['total_energy']:.3f}")
        
        st.info(f"**Energy difference**: {abs(other_result['total_energy'] - calc_result['total_energy']):.3f} between the two patterns for this quantum state.")

    # Analyze sparsity pattern
    def analyze_sparsity_pattern(H, title):
        """Analyze the sparsity pattern of the Hamiltonian"""
        # Find non-zero off-diagonal elements
        H_offdiag = H - np.diag(np.diag(H))
        nonzero_positions = np.where(np.abs(H_offdiag) > 1e-10)
        
        st.write(f"**{title} Sparsity Analysis:**")
        st.write(f"• Total matrix elements: {H.size}")
        st.write(f"• Non-zero diagonal elements: {np.count_nonzero(np.diag(H))}")
        st.write(f"• Non-zero off-diagonal elements: {len(nonzero_positions[0])}")
        st.write(f"• Off-diagonal connections: {len(nonzero_positions[0])//2} (symmetric)")
        
        # Show some off-diagonal connections
        if len(nonzero_positions[0]) > 0:
            st.write("**Example off-diagonal connections:**")
            for i in range(min(5, len(nonzero_positions[0]))):
                row, col = nonzero_positions[0][i], nonzero_positions[1][i]
                if row < col:  # Only show upper triangle
                    state1 = ''.join(map(str, basis_states[row]))
                    state2 = ''.join(map(str, basis_states[col]))
                    value = H[row, col]
                    st.write(f"  |{state1}⟩ ↔ |{state2}⟩: {value:.3f}")
    
    col_sparse1, col_sparse2 = st.columns(2)
    
    with col_sparse1:
        analyze_sparsity_pattern(H1, pattern1)
        
    with col_sparse2:
        analyze_sparsity_pattern(H2, pattern2)
    
    # Physical interpretation of matrix structure
    st.markdown("""
    <div class="physics-insight">
    <h4>🔬 Physical Interpretation of Matrix Structure</h4>
    <p><strong>Diagonal Elements (ZZ terms):</strong></p>
    <ul>
        <li>Represent energy of each basis state</li>
        <li>Determined by spin alignment in coupled pairs</li>
        <li>Different K-patterns create different diagonal structures</li>
    </ul>
    <p><strong>Off-diagonal Elements (XX+YY terms):</strong></p>
    <ul>
        <li>Represent quantum transitions between basis states</li>
        <li>Only connect states that differ by one spin flip in coupled pairs</li>
        <li>Create the quantum dynamics and coherent oscillations</li>
    </ul>
    <p><strong>Matrix Differences:</strong></p>
    <ul>
        <li>Gradient patterns: Structured diagonal variations</li>
        <li>Symmetric patterns: Regular diagonal patterns</li>
        <li>The diagonal structure determines the spectral properties!</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"Error calculating Hamiltonian matrices: {e}")
    st.info("The Hamiltonian calculation requires the full quantum simulation setup.")

# ==========================================
# EIGENVALUE ANALYSIS
# ==========================================

st.markdown("#### ⚛️ Theoretical Eigenvalue Analysis")

# Calculate ZZ eigenvalues analytically
def calculate_zz_eigenvalues(k_01, k_23, k_45):
    """Calculate ZZ Hamiltonian eigenvalues analytically"""
    eigenvalues = []
    # For each combination of spin alignments
    for s01 in [-1, 1]:
        for s23 in [-1, 1]:
            for s45 in [-1, 1]:
                energy = k_01 * s01 + k_23 * s23 + k_45 * s45
                eigenvalues.append(energy)
    return sorted(eigenvalues)

# Get ZZ eigenvalues for both patterns
zz_eigen1 = calculate_zz_eigenvalues(metrics1['k_01'], metrics1['k_23'], metrics1['k_45'])
zz_eigen2 = calculate_zz_eigenvalues(metrics2['k_01'], metrics2['k_23'], metrics2['k_45'])

# Plot eigenvalue comparison
fig_eigen_comp = make_subplots(
    rows=1, cols=2,
    subplot_titles=[f'{pattern1} ZZ Eigenvalues', f'{pattern2} ZZ Eigenvalues']
)

# Pattern 1 eigenvalues
fig_eigen_comp.add_trace(
    go.Scatter(x=list(range(len(zz_eigen1))), y=zz_eigen1,
              mode='lines+markers', name=pattern1,
              line=dict(color='blue', width=3)),
    row=1, col=1
)

# Pattern 2 eigenvalues  
fig_eigen_comp.add_trace(
    go.Scatter(x=list(range(len(zz_eigen2))), y=zz_eigen2,
              mode='lines+markers', name=pattern2,
              line=dict(color='red', width=3)),
    row=1, col=2
)

fig_eigen_comp.update_layout(
    title="ZZ Hamiltonian Eigenvalue Structure Comparison",
    height=400
)

st.plotly_chart(fig_eigen_comp, use_container_width=True)

# Eigenvalue statistics - ensure consistent formatting
col_eigen1, col_eigen2 = st.columns(2)

with col_eigen1:
    st.markdown(f"**{pattern1} Eigenvalues:**")
    st.write(f"Range: {min(zz_eigen1):.3f} to {max(zz_eigen1):.3f}")
    st.write(f"Span: {max(zz_eigen1) - min(zz_eigen1):.3f}")
    st.write(f"Unique values: {len(set(zz_eigen1))}")

with col_eigen2:
    st.markdown(f"**{pattern2} Eigenvalues:**")
    st.write(f"Range: {min(zz_eigen2):.3f} to {max(zz_eigen2):.3f}")
    st.write(f"Span: {max(zz_eigen2) - min(zz_eigen2):.3f}")
    st.write(f"Unique values: {len(set(zz_eigen2))}")

# ==========================================
# TRANSITION FREQUENCY ANALYSIS
# ==========================================

st.markdown("#### 🌊 Transition Frequency Analysis")

# Calculate transition frequencies
def calculate_transition_frequencies(eigenvalues):
    """Calculate all possible transition frequencies"""
    unique_eigenvals = sorted(list(set(eigenvalues)))
    transitions = []
    for i in range(len(unique_eigenvals)):
        for j in range(i + 1, len(unique_eigenvals)):
            freq = abs(unique_eigenvals[j] - unique_eigenvals[i])
            if freq > 0:
                transitions.append(freq)
    return sorted(transitions)

trans1 = calculate_transition_frequencies(zz_eigen1)
trans2 = calculate_transition_frequencies(zz_eigen2)

# Calculate transition statistics
trans1_mean = np.mean(trans1) if trans1 else 0
trans1_var = np.var(trans1) if trans1 else 0
trans2_mean = np.mean(trans2) if trans2 else 0
trans2_var = np.var(trans2) if trans2 else 0

# Display transition analysis with error handling
col_trans1, col_trans2 = st.columns(2)

with col_trans1:
    st.markdown(f"**{pattern1} Transitions:**")
    st.metric("Number of transitions", len(trans1))
    st.metric("Mean frequency", f"{trans1_mean:.3f}" if trans1_mean > 0 else "0.000")
    st.metric("Variance", f"{trans1_var:.3f}" if trans1_var > 0 else "0.000")
    
with col_trans2:
    st.markdown(f"**{pattern2} Transitions:**")
    st.metric("Number of transitions", len(trans2))
    st.metric("Mean frequency", f"{trans2_mean:.3f}" if trans2_mean > 0 else "0.000")
    st.metric("Variance", f"{trans2_var:.3f}" if trans2_var > 0 else "0.000")

# Transition frequency histogram
if trans1 and trans2:
    fig_trans = go.Figure()
    
    fig_trans.add_trace(go.Histogram(
        x=trans1, name=pattern1, opacity=0.7,
        marker_color='blue', nbinsx=20
    ))
    
    fig_trans.add_trace(go.Histogram(
        x=trans2, name=pattern2, opacity=0.7,
        marker_color='red', nbinsx=20
    ))
    
    fig_trans.update_layout(
        title="Transition Frequency Distribution",
        xaxis_title="Transition Frequency",
        yaxis_title="Count",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_trans, use_container_width=True)

# ==========================================
# ENERGY LANDSCAPE VISUALIZATION
# ==========================================

st.markdown("#### 🏔️ Energy Landscape Visualization")

# Create 3D energy landscape visualization
def create_energy_landscape(k_01, k_23, k_45, title):
    """Create a 3D visualization of the energy landscape"""
    
    # Create a grid of ZZ interaction energies
    x = np.linspace(-1, 1, 20)  # s01 values
    y = np.linspace(-1, 1, 20)  # s23 values
    
    # Calculate energies for a 2D slice (fix s45 = 0)
    X, Y = np.meshgrid(x, y)
    Z = k_01 * X + k_23 * Y + k_45 * 0  # Fix s45 = 0
    
    fig_landscape = go.Figure(data=[
        go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', name=title)
    ])
    
    fig_landscape.update_layout(
        title=f"{title} Energy Landscape (s45=0 slice)",
        scene=dict(
            xaxis_title="s01",
            yaxis_title="s23", 
            zaxis_title="Energy"
        ),
        height=500
    )
    
    return fig_landscape

# Create energy landscapes for both patterns
col_land1, col_land2 = st.columns(2)

with col_land1:
    fig_land1 = create_energy_landscape(metrics1['k_01'], metrics1['k_23'], metrics1['k_45'], pattern1)
    st.plotly_chart(fig_land1, use_container_width=True)
    
with col_land2:
    fig_land2 = create_energy_landscape(metrics2['k_01'], metrics2['k_23'], metrics2['k_45'], pattern2)
    st.plotly_chart(fig_land2, use_container_width=True)

# ==========================================
# PHYSICAL INTERPRETATION
# ==========================================

st.markdown("#### 🔬 Physical Interpretation")

# Determine pattern types
is_gradient1 = abs(metrics1['k_gradient']) > 0.5
is_gradient2 = abs(metrics2['k_gradient']) > 0.5

# Calculate variance ratio safely
if trans1_var > 0:
    trans_var_ratio = trans2_var / trans1_var
elif trans2_var > 0:
    trans_var_ratio = float('inf')
else:
    trans_var_ratio = 1.0

# Define ratio display text immediately after calculation
ratio_display = f"{trans_var_ratio:.1f}x" if trans_var_ratio != float('inf') else "∞x"

interpretation_html = f"""
<div class="physics-insight">
<h4>🎯 Why {pattern1} {'(Gradient)' if is_gradient1 else '(Symmetric)'} vs {pattern2} {'(Gradient)' if is_gradient2 else '(Symmetric)'} Differ</h4>
"""

if is_gradient1 and not is_gradient2:
    interpretation_html += f"""
    <p><strong>{pattern1} (Gradient Pattern):</strong></p>
    <ul>
        <li>Creates smooth energy landscapes like 'quantum slides'</li>
        <li>K-gradient: {metrics1['k_gradient']:.2f} (monotonic progression)</li>
        <li>Transition frequency variance: {trans1_var:.3f} (structured)</li>
        <li>Enables uniform probability distribution across eigenstates</li>
        <li>More predictable for ML algorithms</li>
    </ul>
    
    <p><strong>{pattern2} (Symmetric Pattern):</strong></p>
    <ul>
        <li>Creates energy barriers like 'quantum mirrors'</li>
        <li>K-gradient: {metrics2['k_gradient']:.2f} (symmetric/flat)</li>
        <li>Transition frequency variance: {trans2_var:.3f} ({trans_var_ratio:.1f}x {'higher' if trans_var_ratio > 1 else 'lower'})</li>
        <li>Causes localized hot spots and interference patterns</li>
        <li>Less predictable for ML algorithms</li>
    </ul>
    """
elif is_gradient2 and not is_gradient1:
    interpretation_html += f"""
    <p><strong>{pattern2} (Gradient Pattern):</strong></p>
    <ul>
        <li>Creates smooth energy landscapes like 'quantum slides'</li>
        <li>K-gradient: {metrics2['k_gradient']:.2f} (monotonic progression)</li>
        <li>More predictable for ML algorithms</li>
    </ul>
    
    <p><strong>{pattern1} (Symmetric Pattern):</strong></p>
    <ul>
        <li>Creates energy barriers like 'quantum mirrors'</li>
        <li>K-gradient: {metrics1['k_gradient']:.2f} (symmetric/flat)</li>
        <li>Less predictable for ML algorithms</li>
    </ul>
    """
else:
    interpretation_html += f"""
    <p>Both patterns show similar gradient characteristics:</p>
    <ul>
        <li><strong>{pattern1}:</strong> K-gradient = {metrics1['k_gradient']:.2f}</li>
        <li><strong>{pattern2}:</strong> K-gradient = {metrics2['k_gradient']:.2f}</li>
    </ul>
    """

interpretation_html += f"""
<p><strong>Mathematical Mechanism:</strong></p>
<p>The spectral amplitude for frequency ω is proportional to:</p>
<p><strong>A(ω) ∝ |⟨ψᵢ|ψ₀⟩|² × |⟨ψⱼ|ψ₀⟩|² × f(Eᵢ - Eⱼ)</strong></p>
<p>where |ψ₀⟩ = a|A⟩ + b|B⟩ + c|C⟩ is the initial superposition.</p>

<p><strong>Key Insight:</strong> The spectral variance differences come from how quantum 
probability flows through different K-coupling energy landscapes during time evolution!</p>
</div>
"""

st.markdown(interpretation_html, unsafe_allow_html=True)

# ==========================================
# EXPERIMENTAL PREDICTION
# ==========================================

st.markdown("#### 🔮 Experimental Predictions")

# Display predictions in a simpler format to avoid serialization issues
pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.markdown(f"**{pattern1} Pattern Predictions:**")
    st.write(f"• Type: {'Gradient' if is_gradient1 else 'Symmetric'}")
    st.write(f"• Expected spectral peaks: {'Regular, structured' if is_gradient1 else 'Irregular, chaotic'}")
    st.write(f"• Transition frequency variance: {trans1_var:.3f}")
    st.write(f"• ML predictability: {'HIGH' if is_gradient1 else 'LOW'}")

with pred_col2:
    st.markdown(f"**{pattern2} Pattern Predictions:**")
    st.write(f"• Type: {'Gradient' if is_gradient2 else 'Symmetric'}")
    st.write(f"• Expected spectral peaks: {'Regular, structured' if is_gradient2 else 'Irregular, chaotic'}")
    st.write(f"• Transition frequency variance: {trans2_var:.3f}")
    st.write(f"• ML predictability: {'HIGH' if is_gradient2 else 'LOW'}")

# Variance ratio comparison
if trans1_var > 0 and trans2_var > 0:
    ratio_text = f"{trans_var_ratio:.1f}x difference in transition frequency variance"
elif trans1_var == 0 and trans2_var == 0:
    ratio_text = "Both patterns have zero variance (identical transition frequencies)"
else:
    ratio_text = f"One pattern has zero variance (ratio = {ratio_display})"

st.info(f"**Variance Ratio:** {ratio_text}")

# Note about running simulations
if 'diagnostic_results' in st.session_state:
    st.success("✅ Simulation results available! The theoretical predictions above can be compared with the experimental data from your simulations.")
else:
    st.info("💡 Run the K-Pattern Analysis above to see how these theoretical predictions compare with actual simulation results!")

# ==========================================
# SUMMARY TABLE
# ==========================================

st.markdown("#### 📋 Mathematical Summary")

# Create comprehensive summary - ensure all values are strings
summary_data = {
    'Metric': [
        'K-coupling Type',
        'K Gradient',
        'K Symmetry',
        'K Variance',
        'ZZ Eigenvalue Range',
        'Unique Energy Levels',
        'Transition Frequencies',
        'Transition Variance',
        'Predicted ML Performance'
    ],
    pattern1: [
        'Gradient' if is_gradient1 else 'Symmetric',
        f"{metrics1['k_gradient']:.2f}",
        f"{metrics1['k_symmetry']:.2f}",
        f"{metrics1['k_variance']:.3f}",
        f"{max(zz_eigen1) - min(zz_eigen1):.2f}",
        f"{len(set(zz_eigen1))}",
        f"{len(trans1)}",
        f"{trans1_var:.3f}",
        'HIGH' if is_gradient1 else 'LOW'
    ],
    pattern2: [
        'Gradient' if is_gradient2 else 'Symmetric',
        f"{metrics2['k_gradient']:.2f}",
        f"{metrics2['k_symmetry']:.2f}",
        f"{metrics2['k_variance']:.3f}",
        f"{max(zz_eigen2) - min(zz_eigen2):.2f}",
        f"{len(set(zz_eigen2))}",
        f"{len(trans2)}",
        f"{trans2_var:.3f}",
        'HIGH' if is_gradient2 else 'LOW'
    ]
}

df_summary = pd.DataFrame(summary_data)

# Style function for highlighting differences
def highlight_math_differences(data):
    """Highlight the key differences"""
    attr = 'background-color: {}'.format
    
    if data.name == 'K-coupling Type':
        return [attr('lightgreen') if val == 'Gradient' else attr('lightcoral') for val in data]
    elif data.name == 'Predicted ML Performance':
        return [attr('lightgreen') if val == 'HIGH' else attr('lightcoral') for val in data]
    elif data.name == 'Transition Variance':
        # Handle string values properly
        try:
            vals = [float(val) for val in data[1:]]
            if len(vals) == 2 and vals[0] != vals[1]:
                min_val = min(vals)
                return [''] + [attr('lightgreen') if float(val) == min_val else attr('lightcoral') for val in data[1:]]
        except (ValueError, TypeError):
            pass
    
    return ['' for _ in data]

# Style and display with error handling
try:
    styled_df = df_summary.style.apply(highlight_math_differences, axis=1)
    st.dataframe(styled_df, use_container_width=True)
except Exception as e:
    st.warning(f"Display issue with summary table, showing plain version: {e}")
    # Fallback to plain dataframe
    st.dataframe(df_summary, use_container_width=True)

# Final mathematical conclusion with safe ratio display
ratio_comparison = 'lower' if trans_var_ratio > 1 else 'higher' if trans_var_ratio < 1 else 'equal'

st.markdown(f"""
<div class="physics-insight">
<h4>🎯 Mathematical Conclusion</h4>
<p>The theoretical analysis shows that <strong>{pattern1}</strong> has 
{ratio_display} {ratio_comparison} transition frequency variance 
than <strong>{pattern2}</strong>.</p>

<p>This mathematical difference in energy landscape structure predicts that 
{'gradient' if is_gradient1 else 'symmetric'} patterns should be more predictable for ML algorithms 
because they create more regular quantum dynamics.</p>

<p><strong>The key insight:</strong> Gradient K-couplings create smoother energy landscapes that 
lead to more structured transition frequencies, while symmetric patterns create energy barriers 
that cause irregular spectral signatures.</p>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("**🔬 Diagnostic Summary**: This app reveals why certain k-coupling patterns create quantum dynamics that are inherently more difficult for machine learning to invert.")

# Add download button for results if available
if 'diagnostic_results' in st.session_state:
    st.markdown("### 💾 Export Results")
    
    # Create summary dataframe
    export_data = []
    for name, result in st.session_state.diagnostic_results.items():
        export_data.append({
            'pattern': name,
            'k01': result['k_pattern'][(0,1)],
            'k23': result['k_pattern'][(2,3)],
            'k45': result['k_pattern'][(4,5)],
            'total_k': result['total_k'],
            'k_variance': result['k_variance'],
            'q4_peaks': result['fft_results'][4]['n_peaks'],
            'q4_oscillation': np.max(result['probs'][:, 4]) - np.min(result['probs'][:, 4])
        })
    
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False)
    
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="k_pattern_analysis_results.csv",
        mime="text/csv"
    )