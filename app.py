"""
Streamlit UI for Lightning Network Fee Forecasting with FLAML
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from ln_flaml import (
    load_and_filter_channel_updates,
    compute_effective_fee_and_lags,
    select_best_channel,
    train_and_forecast,
)

# Page config
st.set_page_config(
    page_title="⚡ LN Fee Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #f39c12, #e67e22);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #f39c12, #e67e22);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header - will be updated after task selection
header_placeholder = st.empty()
subheader_placeholder = st.empty()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Task Selection")
    
    task = st.selectbox(
        "Choose Prediction Task",
        [
            "Fee Forecasting",
            "Route Optimization",
            "Liquidity Management",
            "Channel Rebalancing",
            "Payment Success Prediction",
            "Network Anomaly Detection"
        ],
        help="Select which LN node management task to optimize"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    data_file = st.text_input(
        "Data File Path",
        value="data/ln_node_data.json",
        help="Path to your LN node gossip data JSON file"
    )
    
    amount_sats = st.number_input(
        "Payment Amount (sats)",
        min_value=1000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Amount in satoshis to calculate routing fees for"
    )
    
    time_budget = st.slider(
        "Training Time Budget (seconds)",
        min_value=10,
        max_value=120,
        value=30,
        step=10,
        help="How long FLAML should search for the best model"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About")
    
    # Task-specific about text
    about_text = {
        "Fee Forecasting": """
        This tool uses **FLAML** (Fast and Lightweight AutoML) to forecast 
        Lightning Network routing fees based on historical channel updates.
        
        **Features:**
        - Automatic model selection
        - Time-series forecasting
        - Channel policy analysis
        """,
        "Route Optimization": """
        Find optimal payment routes by analyzing:
        - Historical success rates
        - Fee structures
        - Channel capacities
        - Network topology
        """,
        "Liquidity Management": """
        Predict and optimize channel liquidity:
        - Forecast liquidity needs
        - Balance allocation strategies
        - Rebalancing recommendations
        """,
        "Channel Rebalancing": """
        Determine when and how to rebalance:
        - Cost-benefit analysis
        - Optimal rebalancing timing
        - Multi-channel coordination
        """,
        "Payment Success Prediction": """
        Predict payment success probability:
        - Route reliability analysis
        - Historical success patterns
        - Real-time network state
        """,
        "Network Anomaly Detection": """
        Detect unusual network behavior:
        - Fee spikes
        - Channel failures
        - Routing anomalies
        """
    }
    
    st.markdown(about_text.get(task, about_text["Fee Forecasting"]))

# Update header based on selected task
task_headers = {
    "Fee Forecasting": ("⚡ Lightning Network Fee Forecaster", "Predict routing fees with FLAML AutoML"),
    "Route Optimization": ("🗺️ Route Optimization", "Find optimal payment paths through the network"),
    "Liquidity Management": ("💧 Liquidity Management", "Optimize channel liquidity allocation"),
    "Channel Rebalancing": ("⚖️ Channel Rebalancing", "Smart rebalancing strategies with ML"),
    "Payment Success Prediction": ("✅ Payment Success Predictor", "Forecast payment success probability"),
    "Network Anomaly Detection": ("🔍 Network Anomaly Detection", "Detect unusual network patterns")
}

header, subheader = task_headers.get(task, task_headers["Fee Forecasting"])
header_placeholder.markdown(f'<p class="main-header">{header}</p>', unsafe_allow_html=True)
subheader_placeholder.markdown(f'<p class="sub-header">{subheader}</p>', unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Show task-specific info
    if task != "Fee Forecasting":
        st.warning(f"🚧 **{task}** - Work in Progress")
        st.markdown("""
        This feature is coming soon! Currently available tasks:
        
        ✅ **Fee Forecasting** - Predict routing fees using historical data
        
        🔜 **Coming Soon:**
        - Route Optimization
        - Liquidity Management
        - Channel Rebalancing
        - Payment Success Prediction
        - Network Anomaly Detection
        """)
        
        # Show task description
        task_descriptions = {
            "Route Optimization": "Optimize payment routes to minimize fees and maximize success rates",
            "Liquidity Management": "Predict and manage channel liquidity needs",
            "Channel Rebalancing": "Determine optimal rebalancing strategies",
            "Payment Success Prediction": "Predict payment success probability before sending",
            "Network Anomaly Detection": "Detect unusual patterns in network activity"
        }
        
        if task in task_descriptions:
            st.info(f"**What this will do:** {task_descriptions[task]}")
    
    elif st.button("🚀 Train & Forecast", type="primary"):
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load data
            status_text.markdown("**Step 1/4:** Loading channel updates...")
            progress_bar.progress(25)
            df = load_and_filter_channel_updates(data_file)
            
            # Display data stats
            st.success(f"✅ Loaded {len(df)} channel updates")
            
            # Step 2: Feature engineering
            status_text.markdown("**Step 2/4:** Computing effective fees and lag features...")
            progress_bar.progress(50)
            df = compute_effective_fee_and_lags(df, amount_sats)
            
            # Step 3: Select channel
            status_text.markdown("**Step 3/4:** Selecting best channel...")
            progress_bar.progress(60)
            channel_id, channel_df = select_best_channel(df)
            
            st.info(f"📡 Selected channel: **{channel_id}** with {len(channel_df)} valid samples")
            
            # Step 4: Train model
            status_text.markdown("**Step 4/4:** Training FLAML model...")
            progress_bar.progress(75)
            
            start_time = time.time()
            results = train_and_forecast(channel_df, time_budget)
            training_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.markdown("**✨ Training complete!**")
            
            # Results section
            st.markdown("---")
            st.markdown("## 📈 Forecast Results")
            
            # Metrics row
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric(
                    "Test MAE",
                    f"{results['mae']:.2f} msat",
                    help="Mean Absolute Error on test set"
                )
            
            with col_b:
                st.metric(
                    "Training Time",
                    f"{training_time:.1f}s",
                    help="Time spent training the model"
                )
            
            with col_c:
                st.metric(
                    "Test Samples",
                    len(results['y_test']),
                    help="Number of test predictions"
                )
            
            with col_d:
                avg_error_pct = (results['mae'] / results['y_test'].mean()) * 100
                st.metric(
                    "Avg Error %",
                    f"{avg_error_pct:.1f}%",
                    help="MAE as percentage of mean actual fee"
                )
            
            # Model info
            st.markdown("### 🤖 Best Model")
            with st.expander("Show model details"):
                st.code(results['best_model'], language='python')
            
            # Predictions visualization
            st.markdown("### 📊 Predictions vs Actuals")
            
            test_df = results['test_df'].reset_index(drop=True)
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Predicted vs Actual Fees', 'Prediction Error'),
                vertical_spacing=0.12,
                row_heights=[0.7, 0.3]
            )
            
            # Line plot for predictions vs actuals
            indices = list(range(len(y_test)))
            
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_test,
                    name='Actual',
                    mode='lines+markers',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=10)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_pred,
                    name='Predicted',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=3, dash='dash'),
                    marker=dict(size=10, symbol='diamond')
                ),
                row=1, col=1
            )
            
            # Error bars
            errors = y_test - y_pred
            fig.add_trace(
                go.Bar(
                    x=indices,
                    y=errors,
                    name='Error',
                    marker=dict(
                        color=errors,
                        colorscale='RdYlGn',
                        reversescale=True,
                        line=dict(width=1, color='white')
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Test Sample Index", row=2, col=1)
            fig.update_yaxes(title_text="Effective Fee (msat)", row=1, col=1)
            fig.update_yaxes(title_text="Error (msat)", row=2, col=1)
            
            fig.update_layout(
                height=700,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed predictions table
            st.markdown("### 📋 Detailed Predictions")
            
            pred_df = pd.DataFrame({
                'Sample': range(len(y_test)),
                'Timestamp': test_df['timestamp'].values,
                'Actual Fee (msat)': y_test.round(2),
                'Predicted Fee (msat)': y_pred.round(2),
                'Error (msat)': (y_test - y_pred).round(2),
                'Error %': ((y_test - y_pred) / y_test * 100).round(2)
            })
            
            # Color-code rows based on error magnitude
            def color_error(val):
                if abs(val) < 1000:
                    return 'background-color: #d5f4e6'  # light green
                elif abs(val) < 5000:
                    return 'background-color: #fff4cc'  # light yellow
                else:
                    return 'background-color: #ffe0e0'  # light red
            
            styled_df = pred_df.style.applymap(
                color_error, 
                subset=['Error (msat)']
            ).format({
                'Actual Fee (msat)': '{:.2f}',
                'Predicted Fee (msat)': '{:.2f}',
                'Error (msat)': '{:.2f}',
                'Error %': '{:.2f}'
            })
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            st.markdown("### 📈 Error Statistics")
            col_e, col_f, col_g, col_h = st.columns(4)
            
            with col_e:
                st.metric("Min Error", f"{errors.min():.2f} msat")
            with col_f:
                st.metric("Max Error", f"{errors.max():.2f} msat")
            with col_g:
                st.metric("Std Dev", f"{errors.std():.2f} msat")
            with col_h:
                st.metric("RMSE", f"{(errors**2).mean()**0.5:.2f} msat")
            
            # Download results
            st.markdown("### 💾 Export Results")
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"ln_fee_predictions_{channel_id}.csv",
                mime="text/csv",
            )
            
        except FileNotFoundError:
            st.error(f"❌ File not found: {data_file}")
            st.info("Please check the file path in the sidebar configuration.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    else:
        # Welcome message when not training
        st.markdown(f"""
        ### 👋 Welcome to {task}!
        
        Click the **Train & Forecast** button above to start the analysis.
        
        **What this does:**
        1. 📂 Loads your LN channel gossip data
        2. 🔧 Extracts routing patterns
        3. 🤖 Trains an AutoML model with FLAML
        4. 📊 Generates predictions and insights
        
        Adjust the configuration in the sidebar to customize the analysis.
        """)
        
        # Show sample visualization
        st.markdown("---")
        st.markdown("### 📊 Sample Workflow")
        
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            st.markdown("#### 1️⃣ Configure")
            st.markdown("""
            - Set payment amount
            - Choose time budget
            - Select data file
            """)
        
        with col_y:
            st.markdown("#### 2️⃣ Train")
            st.markdown("""
            - Load & clean data
            - Engineer features
            - Train FLAML model
            """)
        
        with col_z:
            st.markdown("#### 3️⃣ Analyze")
            st.markdown("""
            - View predictions
            - Check accuracy
            - Export results
            """)
        
        # Show all available tasks
        st.markdown("---")
        st.markdown("### 🎯 All Available Tasks")
        
        task_col1, task_col2 = st.columns(2)
        
        with task_col1:
            st.success("✅ **Fee Forecasting** - Available Now")
            st.markdown("Predict routing fees using time-series analysis")
            
            st.info("🔜 **Route Optimization** - Coming Soon")
            st.markdown("Find optimal payment paths")
            
            st.info("🔜 **Liquidity Management** - Coming Soon")
            st.markdown("Optimize channel liquidity")
        
        with task_col2:
            st.info("🔜 **Channel Rebalancing** - Coming Soon")
            st.markdown("Smart rebalancing strategies")
            
            st.info("🔜 **Payment Success Prediction** - Coming Soon")
            st.markdown("Forecast payment probability")
            
            st.info("🔜 **Network Anomaly Detection** - Coming Soon")
            st.markdown("Detect unusual patterns")

