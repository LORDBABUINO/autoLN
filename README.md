# Lightning Network Fee Forecasting with FLAML

Integration of Lightning Development Kit (LDK) with FLAML AutoML for optimizing Bitcoin Lightning node management.

## Overview

This project uses FLAML (Fast and Lightweight AutoML) to forecast routing fees on the Lightning Network. It analyzes channel update messages from LN gossip data and predicts future effective fees for payment routing.

## Features

- Parses Lightning Network gossip data (channel_update messages)
- Computes effective routing fees for configurable payment amounts
- Automatically selects the channel with most historical data
- Uses time-series lag features (lag1, lag2, lag3) for forecasting
- Trains FLAML AutoML regression models with minimal configuration
- Provides MAE (Mean Absolute Error) metrics and predictions

## Installation

This project uses `uv` for package management:

```bash
# Dependencies are already configured in pyproject.toml
uv sync
```

## Usage

### Streamlit Web UI (Recommended)

Launch the beautiful web interface:

```bash
uv run streamlit run app.py
```

Then open your browser to `http://localhost:8501`

Features:

- **Multiple Task Selection** - Choose from 6 different LN optimization tasks
- Interactive configuration sidebar
- Real-time progress tracking
- Beautiful visualizations with Plotly
- Prediction vs actual charts
- Error analysis
- CSV export of results

#### Available Tasks

- ✅ **Fee Forecasting** - Predict routing fees (fully implemented)
- 🔜 **Route Optimization** - Coming soon
- 🔜 **Liquidity Management** - Coming soon
- 🔜 **Channel Rebalancing** - Coming soon
- 🔜 **Payment Success Prediction** - Coming soon
- 🔜 **Network Anomaly Detection** - Coming soon

### CLI (Command Line)

Run the fee forecasting with default settings:

```bash
uv run python main.py
```

#### CLI Arguments

- `--data`: Path to LN node data JSON file (default: `data/ln_node_data.json`)
- `--amount_sats`: Payment amount in satoshis for fee calculation (default: `100000`)
- `--time_budget`: FLAML training time budget in seconds (default: `30`)

### Examples

```bash
# Forecast fees for 50,000 sat payments with 60s training
uv run python main.py --amount_sats 50000 --time_budget 60

# Use custom data file
uv run python main.py --data custom_data.json
```

## Data Format

The input JSON should contain Lightning Network gossip messages. The tool specifically uses `channel_update` messages (type 258) which include:

- `short_channel_id`: Channel identifier
- `timestamp`: Unix timestamp of the update
- `fee_base_msat`: Base fee in millisatoshis
- `fee_proportional_millionths`: Proportional fee rate
- `cltv_expiry_delta`, `channel_flags`, `htlc_minimum_msat`, `htlc_maximum_msat`: Other routing policy parameters

## Model Details

- **Task**: Regression (forecasting next-step effective fees)
- **Features**: 3 time-series lags + channel policy parameters (7 features total)
- **Metric**: MAE (Mean Absolute Error)
- **Models**: FLAML automatically selects from LightGBM, XGBoost, Random Forest, Extra Trees
- **Training**: Time-ordered split (80/20), no shuffling to preserve temporal structure

### Effective Fee Calculation

For a payment amount `A` (in millisatoshis):

```
effective_fee = fee_base_msat + (A × fee_proportional_millionths / 1,000,000)
```

## Output

The tool prints:

- Selected channel and number of updates
- Training/test split sizes
- Best model found by FLAML
- Test MAE
- Last 5 predictions vs actual values with errors

## Project Structure

```
.
├── app.py               # Streamlit web UI (recommended)
├── main.py              # CLI entry point
├── ln_flaml.py          # Core forecasting logic
├── data/
│   └── ln_node_data.json  # Sample LN gossip data
├── pyproject.toml       # Dependencies
├── .cursorrules         # Project-specific rules
└── README.md            # This file
```

## License

MIT
