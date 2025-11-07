# 🎯 Task Selection Feature

## Overview

The Streamlit app now includes a dropdown menu to select different Lightning Network node management tasks. Currently, only **Fee Forecasting** is fully implemented, while other tasks show a "Work in Progress" message.

## Available Tasks

### ✅ Fee Forecasting (Implemented)
- **What it does**: Predicts routing fees using historical channel update data
- **Features**: Time-series analysis with lag features, AutoML model selection
- **Output**: Fee predictions, accuracy metrics, interactive charts

### 🔜 Route Optimization (Coming Soon)
- **Description**: Find optimal payment routes through the network
- **Will analyze**: Historical success rates, fee structures, channel capacities, network topology

### 🔜 Liquidity Management (Coming Soon)
- **Description**: Predict and optimize channel liquidity allocation
- **Will analyze**: Forecast liquidity needs, balance allocation strategies, rebalancing recommendations

### 🔜 Channel Rebalancing (Coming Soon)
- **Description**: Determine when and how to rebalance channels
- **Will analyze**: Cost-benefit analysis, optimal timing, multi-channel coordination

### 🔜 Payment Success Prediction (Coming Soon)
- **Description**: Forecast payment success probability before sending
- **Will analyze**: Route reliability, historical patterns, real-time network state

### 🔜 Network Anomaly Detection (Coming Soon)
- **Description**: Detect unusual network behavior
- **Will analyze**: Fee spikes, channel failures, routing anomalies

## User Experience

### Task Selection
1. Open the sidebar in the Streamlit app
2. Use the **"Choose Prediction Task"** dropdown at the top
3. Select your desired task

### For Implemented Tasks (Fee Forecasting)
- Full functionality with "Train & Forecast" button
- Configuration options in sidebar
- Complete results with visualizations

### For Coming Soon Tasks
- Yellow warning badge: "🚧 Work in Progress"
- List of currently available vs coming features
- Task-specific description of what it will do
- No training button (prevented from running)

## Dynamic UI Updates

When you select a different task, the UI automatically updates:

1. **Header** - Changes to task-specific title with icon
   - ⚡ Fee Forecaster
   - 🗺️ Route Optimization
   - 💧 Liquidity Management
   - ⚖️ Channel Rebalancing
   - ✅ Payment Success Predictor
   - 🔍 Network Anomaly Detection

2. **Subheader** - Updates with task-specific description

3. **Sidebar About Section** - Shows task-specific features and capabilities

4. **Main Content** - Either shows:
   - Training button (for Fee Forecasting)
   - Work in Progress message (for other tasks)

5. **Welcome Screen** - Displays all tasks with status badges

## Code Structure

### Task Configuration
```python
# Task dropdown in sidebar
task = st.selectbox(
    "Choose Prediction Task",
    [
        "Fee Forecasting",
        "Route Optimization",
        "Liquidity Management",
        "Channel Rebalancing",
        "Payment Success Prediction",
        "Network Anomaly Detection"
    ]
)
```

### Conditional Rendering
```python
if task != "Fee Forecasting":
    # Show work in progress message
    st.warning(f"🚧 **{task}** - Work in Progress")
else:
    # Show training button and full functionality
    if st.button("🚀 Train & Forecast"):
        # Run training pipeline
```

## Future Expansion

To add a new task implementation:

1. Update the conditional in `app.py` to include the new task
2. Create task-specific data loading and feature engineering
3. Add task-specific model training logic
4. Design task-specific visualizations
5. Update documentation

## Benefits

- **Extensible**: Easy to add new tasks
- **Clear UX**: Users know what's available vs coming soon
- **Professional**: Shows the roadmap and planned features
- **Educational**: Each task includes descriptions of what it will do
- **Maintainable**: Centralized task configuration makes updates easy

