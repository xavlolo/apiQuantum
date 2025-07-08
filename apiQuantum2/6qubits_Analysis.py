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
    """Simple FFT analysis to find frequency peaks"""
    # Check if there's any oscillation
    if np.std(probs[:, qubit_idx]) < 0.001:
        return [], (np.array([0]), np.array([0]))
    
    dt = times[1] - times[0]
    
    # Remove DC component
    sig = probs[:, qubit_idx] - np.mean(probs[:, qubit_idx])
    
    # Apply Hanning window
    win = np.hanning(len(sig))
    sig_windowed = sig * win
    
    # FFT
    fft = np.fft.fft(sig_windowed)
    freq = np.fft.fftfreq(len(fft), dt)
    
    # Keep only positive frequencies
    mask = freq >= 0
    pos_freq = freq[mask]
    pos_mag = np.abs(fft[mask])
    
    # Find peaks
    threshold = 0.1 * np.max(pos_mag) if np.max(pos_mag) > 0 else 0
    peaks, _ = find_peaks(pos_mag, height=threshold, distance=5)
    
    # Skip DC component
    peaks = peaks[pos_freq[peaks] > 0.01]
    
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
                j_coupling, t_max=t_max, debug=debug_mode
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
    
    # FIXED: Changed from 3x5 to 3x6 to accommodate all 6 patterns
    fig = make_subplots(
        rows=3, cols=6,
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
    
    # FIXED: Changed from 3x5 to 3x6 to accommodate all 6 patterns
    fig_fft = make_subplots(
        rows=3, cols=6,
        subplot_titles=list(k_patterns.keys()),
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.05
    )
    
    for col, (name, result) in enumerate(results.items(), 1):
        for row, q in enumerate([0, 2, 4], 1):
            freq, mag = result['fft_results'][q]['fft_data']
            peaks = result['fft_results'][q]['peaks']
            
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
    
    st.markdown('<div class="physics-insight">', unsafe_allow_html=True)
    st.markdown("### 🧠 Key Insights")
    
    # Analyze results
    easy_patterns = ['k_zero_baseline', 'gradient_up', 'gradient_down']
    hard_patterns = ['uniform_1.5', 'well_shallow', 'barrier_low']
    
    st.markdown("#### 🟢 Easy Patterns (Good for ML):")
    for pattern in easy_patterns:
        if pattern in results:
            result = results[pattern]
            st.markdown(f"**{pattern}**: k-values = {list(result['k_pattern'].values())}")
            
            # Analyze signal characteristics
            probs = result['probs']
            q4_osc = np.max(probs[:, 4]) - np.min(probs[:, 4])
            n_peaks_q4 = result['fft_results'][4]['n_peaks']
            
            st.markdown(f"- Q4 oscillation amplitude: {q4_osc:.3f}")
            st.markdown(f"- Q4 number of FFT peaks: {n_peaks_q4}")
            
            if pattern == 'k_zero_baseline':
                st.markdown("- **Physics**: No coupling → free evolution → minimal dynamics")
            else:
                st.markdown("- **Physics**: Smooth gradient → distinct excitation dynamics → unique FFT signatures")
    
    st.markdown("#### 🔴 Hard Patterns (Poor for ML):")
    for pattern in hard_patterns:
        if pattern in results:
            result = results[pattern]
            st.markdown(f"**{pattern}**: k-values = {list(result['k_pattern'].values())}")
            
            probs = result['probs']
            q4_osc = np.max(probs[:, 4]) - np.min(probs[:, 4])
            n_peaks_q4 = result['fft_results'][4]['n_peaks']
            
            st.markdown(f"- Q4 oscillation amplitude: {q4_osc:.3f}")
            st.markdown(f"- Q4 number of FFT peaks: {n_peaks_q4}")
            
            if pattern == 'uniform_1.5':
                st.markdown("- **Physics**: Identical couplings → symmetric dynamics → degenerate frequencies")
            elif pattern == 'well_shallow':
                st.markdown("- **Physics**: Weak center coupling → excitation trapping → limited frequency components")
            elif pattern == 'barrier_low':
                st.markdown("- **Physics**: Strong center barrier → split dynamics → complex interference patterns")
    
    st.markdown("### 🎯 ML Implications")
    st.markdown("""
    **Why some k-patterns are fundamentally harder for ML:**
    
    1. **Information Content**: Gradient patterns encode more distinguishable information in their FFT spectra
    2. **Degeneracy Issues**: Uniform patterns create degenerate eigenvalues → ambiguous frequency signatures
    3. **Nonlinear Mapping**: Wells and barriers create highly nonlinear state-to-spectrum mappings
    4. **Signal Quality**: Some patterns produce weak oscillations that are hard to detect reliably
    
    **Recommendation**: For quantum state reconstruction via ML, use coupling patterns with:
    - Clear gradients or asymmetries
    - Avoided degeneracies
    - Strong signal propagation characteristics
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Store results for further analysis
    st.session_state.diagnostic_results = results
    
else:
    st.info("👆 Click 'Run K-Pattern Analysis' to start the comparative study!")

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