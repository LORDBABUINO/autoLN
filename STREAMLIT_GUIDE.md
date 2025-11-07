# 🎨 Streamlit UI Guide

## Quick Start

Launch the web interface:

```bash
uv run streamlit run app.py
```

Open your browser to: **http://localhost:8501**

## Features

### 📱 Main Interface

1. **Header Section**
   - Beautiful gradient title
   - Quick overview of the tool

2. **Sidebar Configuration**
   - **Data File Path**: Path to your LN gossip JSON
   - **Payment Amount**: Configure the satoshi amount (1k - 10M sats)
   - **Training Time Budget**: Slider for FLAML training time (10-120s)

3. **Main Action Button**
   - Large, centered "Train & Forecast" button
   - Initiates the full ML pipeline

### 📊 Results Display

After clicking "Train & Forecast", you'll see:

#### 1. Progress Tracking
- Real-time progress bar (4 steps)
- Status updates for each pipeline stage:
  - Loading data
  - Feature engineering
  - Channel selection
  - Model training

#### 2. Key Metrics Dashboard
Four metric cards showing:
- **Test MAE**: Mean Absolute Error in millisatoshis
- **Training Time**: Actual time spent training
- **Test Samples**: Number of predictions made
- **Avg Error %**: Error as percentage of mean fee

#### 3. Best Model Display
- Shows the optimal model found by FLAML
- Includes hyperparameters for reproducibility

#### 4. Interactive Visualizations
Two synchronized charts:
- **Top Chart**: Line graph comparing predicted vs actual fees
  - Blue solid line: Actual fees
  - Red dashed line: Predicted fees
  - Interactive hover for values
  
- **Bottom Chart**: Bar chart showing prediction errors
  - Color-coded (green = small error, red = large error)
  - Shows over/under-prediction patterns

#### 5. Detailed Predictions Table
- Scrollable, styled dataframe
- Columns:
  - Sample index
  - Timestamp
  - Actual fee
  - Predicted fee
  - Error (absolute)
  - Error percentage
- Color-coded background gradient on error column

#### 6. Export Functionality
- Download button for CSV export
- Filename includes channel ID for easy tracking

## Color Scheme

- **Primary**: Orange gradient (#f39c12 → #e67e22)
- **Secondary**: Purple gradient (#667eea → #764ba2)
- **Accent**: Blue (#3498db) and Red (#e74c3c)
- **Neutral**: Grays (#7f8c8d)

## Tips

- **Increase time budget** (60-120s) for better model accuracy on larger datasets
- **Lower payment amounts** (10k-50k sats) to see micro-payment fee patterns
- **Higher payment amounts** (500k+ sats) to analyze channel capacity constraints
- Use the **CSV export** to analyze trends over time or compare channels

## Responsive Design

The UI adapts to different screen sizes:
- Wide layout mode for desktop
- Responsive columns for metrics
- Scrollable tables and charts

## Error Handling

The UI gracefully handles:
- Missing data files
- Insufficient training data
- Model training failures
- Invalid configurations

All errors show:
- User-friendly error message
- Expandable details with full traceback
- Suggestions for resolution

