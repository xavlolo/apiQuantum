"""
Quantum FFT State Analyzer
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
import sklearn

# Page configuration MUST BE FIRST ST COMMAND!
st.set_page_config(
    page_title="Quantum FFT State Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# NOW you can use other st commands
st.sidebar.text(f"sklearn: {sklearn.__version__}")

# In the status bar at the bottom
with st.sidebar:
    with st.expander("System Info"):
        st.text(f"sklearn: {sklearn.__version__}")
        st.text(f"numpy: {np.__version__}")
        st.text(f"pandas: {pd.__version__}")
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
            # Placeholder for actual quantum simulation
            # In real implementation, call your quantum simulation functions here
            st.session_state.forward_results = {
                'peaks': [
                    {'freq': 0.523, 'amp': 0.145},
                    {'freq': 0.817, 'amp': 0.098},
                    {'freq': 1.234, 'amp': 0.203}
                ],
                'success': True
            }
    
    # Results section
    if st.session_state.forward_results:
        st.markdown("### 📊 Results")
        st.markdown("**Detected Peaks:**")
        for peak in st.session_state.forward_results['peaks']:
            st.write(f"• {peak['freq']:.3f} Hz (amp: {peak['amp']:.3f})")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            if st.button("📊 Show FFT Plot", key="fft_plot"):
                # Placeholder for FFT plot
                fig, ax = plt.subplots(figsize=(6, 4))
                freqs = np.linspace(0, 2, 1000)
                # Simulated FFT with peaks
                fft_mag = np.zeros_like(freqs)
                for peak in st.session_state.forward_results['peaks']:
                    fft_mag += peak['amp'] * np.exp(-((freqs - peak['freq'])/0.05)**2)
                ax.plot(freqs, fft_mag)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.set_title('FFT of Q4')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with col_plot2:
            if st.button("📈 Show Dynamics", key="dynamics_plot"):
                # Placeholder for dynamics plot
                fig, ax = plt.subplots(figsize=(6, 4))
                t = np.linspace(0, 50, 1000)
                # Simulated dynamics
                dynamics = 0.5 + 0.3*np.sin(2*np.pi*0.523*t) + 0.2*np.sin(2*np.pi*0.817*t)
                ax.plot(t, dynamics)
                ax.set_xlabel('Time')
                ax.set_ylabel('P(1)')
                ax.set_title('Q4 Dynamics')
                ax.grid(True, alpha=0.3)
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
    
    # FFT Peak inputs
    st.markdown("**FFT Peak Frequencies (Hz):**")
    col_freq, col_amp = st.columns(2)
    
    with col_freq:
        st.markdown("**Frequencies:**")
        peak1_freq = st.number_input("Peak 1", value=0.523, min_value=0.0, max_value=5.0, step=0.001, 
                                    format="%.3f", key="p1f")
        peak2_freq = st.number_input("Peak 2", value=0.817, min_value=0.0, max_value=5.0, step=0.001, 
                                    format="%.3f", key="p2f")
        peak3_freq = st.number_input("Peak 3", value=1.234, min_value=0.0, max_value=5.0, step=0.001, 
                                    format="%.3f", key="p3f")
        peak4_freq = st.number_input("Peak 4", value=0.000, min_value=0.0, max_value=5.0, step=0.001, 
                                    format="%.3f", key="p4f")
        peak5_freq = st.number_input("Peak 5", value=0.000, min_value=0.0, max_value=5.0, step=0.001, 
                                    format="%.3f", key="p5f")
    
    with col_amp:
        st.markdown("**Amplitudes:**")
        peak1_amp = st.number_input("Amp 1", value=0.145, min_value=0.0, max_value=1.0, step=0.001, 
                                   format="%.3f", key="p1a")
        peak2_amp = st.number_input("Amp 2", value=0.098, min_value=0.0, max_value=1.0, step=0.001, 
                                   format="%.3f", key="p2a")
        peak3_amp = st.number_input("Amp 3", value=0.203, min_value=0.0, max_value=1.0, step=0.001, 
                                   format="%.3f", key="p3a")
        peak4_amp = st.number_input("Amp 4", value=0.000, min_value=0.0, max_value=1.0, step=0.001, 
                                   format="%.3f", key="p4a")
        peak5_amp = st.number_input("Amp 5", value=0.000, min_value=0.0, max_value=1.0, step=0.001, 
                                   format="%.3f", key="p5a")
    
    # Predict button
    if st.button("🧮 Predict State", key="predict"):
        with st.spinner("Running ML prediction..."):
            # Check if model exists
            if os.path.exists('quantum_inverse_model.pkl'):
                try:
                    # Load model
                    model_data = joblib.load('quantum_inverse_model.pkl')
                    model = model_data['model']
                    scaler = model_data['scaler']
                    
                    # Prepare features
                    features = [peak1_freq, peak1_amp, peak2_freq, peak2_amp, 
                               peak3_freq, peak3_amp, peak4_freq, peak4_amp,
                               peak5_freq, peak5_amp]
                    X_test = np.array([features])
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)[0]
                    
                    st.session_state.inverse_results = {
                        'a_mag': y_pred[0],
                        'b_mag': y_pred[1],
                        'c_mag': y_pred[2],
                        'confidence': 94,  # Placeholder
                        'success': True
                    }
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    # Use placeholder results
                    st.session_state.inverse_results = {
                        'a_mag': 2.47,
                        'b_mag': 1.68,
                        'c_mag': 1.09,
                        'confidence': 94,
                        'success': True
                    }
            else:
                # Placeholder results when model not available
                st.session_state.inverse_results = {
                    'a_mag': 2.47,
                    'b_mag': 1.68,
                    'c_mag': 1.09,
                    'confidence': 94,
                    'success': True
                }
    
    # Results section
    if st.session_state.inverse_results:
        st.markdown("### 📊 Results")
        st.markdown("**Predicted State Magnitudes:**")
        st.write(f"|a| = {st.session_state.inverse_results['a_mag']:.2f}")
        st.write(f"|b| = {st.session_state.inverse_results['b_mag']:.2f}")
        st.write(f"|c| = {st.session_state.inverse_results['c_mag']:.2f}")
        st.write(f"")
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
        st.warning("Status: Ready | Model: Using placeholder (quantum_inverse_model.pkl not found)")
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
    3. Click "Predict State"
    4. View predicted quantum state magnitudes
    
    ### Features
    - 🔒 Lock icon syncs k-values between panels
    - 📊 Interactive plots for visualization
    - 🧮 ML-based state prediction
    
    ### Deployment
    - Save this as `app.py`
    - Install requirements: `streamlit`, `numpy`, `pandas`, `matplotlib`, `plotly`, `joblib`
    - Run locally: `streamlit run app.py`
    - Deploy on Streamlit Cloud, Heroku, or your server
    """)