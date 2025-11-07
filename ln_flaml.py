"""
Lightning Network Fee Forecasting with FLAML
"""
import json

import numpy as np
import pandas as pd
from flaml import AutoML


def load_and_filter_channel_updates(data_path: str) -> pd.DataFrame:
    """Load JSON and filter to channel_update messages only."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Filter for channel_update messages (type 258)
    updates = [msg for msg in data if msg.get('type') == 258]
    
    # Convert to DataFrame
    df = pd.DataFrame(updates)
    
    # Keep only required columns and drop rows with missing critical fields
    required_cols = ['short_channel_id', 'timestamp', 'fee_base_msat', 
                     'fee_proportional_millionths', 'channel_flags', 
                     'cltv_expiry_delta', 'htlc_minimum_msat', 'htlc_maximum_msat']
    
    # Filter columns that exist
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Drop rows with missing values in critical fields
    critical_fields = ['short_channel_id', 'timestamp', 'fee_base_msat', 'fee_proportional_millionths']
    df = df.dropna(subset=critical_fields)
    
    # Ensure numeric types
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['fee_base_msat'] = pd.to_numeric(df['fee_base_msat'], errors='coerce')
    df['fee_proportional_millionths'] = pd.to_numeric(df['fee_proportional_millionths'], errors='coerce')
    df['channel_flags'] = pd.to_numeric(df['channel_flags'], errors='coerce')
    df['cltv_expiry_delta'] = pd.to_numeric(df['cltv_expiry_delta'], errors='coerce')
    df['htlc_minimum_msat'] = pd.to_numeric(df['htlc_minimum_msat'], errors='coerce')
    df['htlc_maximum_msat'] = pd.to_numeric(df['htlc_maximum_msat'], errors='coerce')
    
    # Drop any rows that failed numeric conversion
    df = df.dropna(subset=critical_fields)
    
    # Sort by channel and timestamp
    df = df.sort_values(['short_channel_id', 'timestamp'])
    
    # Drop exact duplicates
    df = df.drop_duplicates()
    
    return df.reset_index(drop=True)


def compute_effective_fee_and_lags(df: pd.DataFrame, amount_sats: int = 100_000) -> pd.DataFrame:
    """Compute effective fee for a fixed amount and create lag features."""
    amount_msat = amount_sats * 1_000
    
    # Compute effective fee
    df['eff_fee_msat'] = (
        df['fee_base_msat'] + 
        (amount_msat * df['fee_proportional_millionths'] / 1_000_000).astype(int)
    )
    
    # Create lag features per channel
    df['lag1'] = df.groupby('short_channel_id')['eff_fee_msat'].shift(1)
    df['lag2'] = df.groupby('short_channel_id')['eff_fee_msat'].shift(2)
    df['lag3'] = df.groupby('short_channel_id')['eff_fee_msat'].shift(3)
    
    return df


def select_best_channel(df: pd.DataFrame, fallback_channel: str = '756252x4532x0', min_updates: int = 13) -> tuple:
    """Select channel with most updates (needs at least min_updates to have enough after lag features)."""
    channel_counts = df['short_channel_id'].value_counts()
    
    if len(channel_counts) == 0:
        raise ValueError("No channels found in data")
    
    # Pick channel with most updates
    best_channel = channel_counts.index[0]
    count = channel_counts.iloc[0]
    
    # If no channel has enough updates, pick the one with the most
    channels_with_enough = channel_counts[channel_counts >= min_updates]
    
    if len(channels_with_enough) > 0:
        # If fallback channel has enough data, prefer it for consistency
        if fallback_channel in channels_with_enough.index:
            best_channel = fallback_channel
            count = channel_counts[fallback_channel]
        else:
            best_channel = channels_with_enough.index[0]
            count = channels_with_enough.iloc[0]
    else:
        print(f"\nWarning: No channel has {min_updates}+ updates. Using channel with most updates.")
    
    print(f"\nSelected channel: {best_channel} with {count} updates")
    
    # Filter to selected channel
    channel_df = df[df['short_channel_id'] == best_channel].copy()
    
    # Drop rows with NaN in lag features (first 3 rows per channel)
    channel_df = channel_df.dropna(subset=['lag1', 'lag2', 'lag3'])
    
    print(f"After removing NaN lags: {len(channel_df)} rows available")
    
    return best_channel, channel_df


def train_and_forecast(df: pd.DataFrame, time_budget: int = 30) -> dict:
    """Train FLAML model and forecast next effective fee."""
    
    if len(df) < 10:
        raise ValueError(f"Not enough data for training. Found {len(df)} rows, need at least 10")
    
    # Feature columns
    feature_cols = ['lag1', 'lag2', 'lag3', 'channel_flags', 
                    'cltv_expiry_delta', 'htlc_minimum_msat', 'htlc_maximum_msat']
    
    # Only use features that exist and have no NaN
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Prepare train/test split (time-ordered, last 20% for test)
    split_idx = int(len(df) * 0.8)
    split_idx = max(split_idx, 1)  # Ensure at least 1 training sample
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # If test set is empty, use last few rows from train for validation
    if len(test_df) == 0:
        split_idx = max(1, len(df) - 3)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['eff_fee_msat']
    
    X_test = test_df[available_features].fillna(0)
    y_test = test_df['eff_fee_msat']
    
    print(f"\nTraining on {len(train_df)} samples, testing on {len(test_df)} samples")
    print(f"Features: {available_features}")
    
    # Train FLAML AutoML
    automl = AutoML()
    
    automl_settings = {
        'time_budget': time_budget,
        'metric': 'mae',
        'task': 'regression',
        'log_file_name': 'flaml_ln.log',
        'verbose': 0,
        'estimator_list': ['lgbm', 'xgboost', 'rf', 'extra_tree'],
    }
    
    print(f"\nTraining FLAML model (budget: {time_budget}s)...")
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    # Make predictions
    y_pred = automl.predict(X_test)
    
    # Compute MAE
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Get best model info
    best_model = automl.model.estimator
    
    results = {
        'mae': mae,
        'best_model': str(best_model),
        'y_test': y_test.values,
        'y_pred': y_pred,
        'test_df': test_df,
        'automl': automl
    }
    
    return results


def print_results(results: dict, channel_id: str):
    """Print forecast results."""
    print("\n" + "="*60)
    print("FORECAST RESULTS")
    print("="*60)
    print(f"\nChannel: {channel_id}")
    print(f"Best model: {results['best_model']}")
    print(f"Test MAE: {results['mae']:.2f} msat")
    
    print("\nLast 5 predictions vs actuals:")
    print("-" * 60)
    print(f"{'Timestamp':<15} {'Actual':<15} {'Predicted':<15} {'Error':<15}")
    print("-" * 60)
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    test_df = results['test_df']
    
    # Show last 5 (or fewer if less available)
    n_show = min(5, len(y_test))
    for i in range(-n_show, 0):
        timestamp = test_df.iloc[i]['timestamp']
        actual = y_test[i]
        predicted = y_pred[i]
        error = actual - predicted
        print(f"{timestamp:<15.0f} {actual:<15.2f} {predicted:<15.2f} {error:<15.2f}")
    
    print("-" * 60)
    print(f"\nMean Absolute Error: {results['mae']:.2f} msat")
    print("="*60)


def run_forecast(data_path: str, amount_sats: int = 100_000, time_budget: int = 30):
    """Main pipeline to run fee forecasting."""
    print(f"Loading data from {data_path}...")
    df = load_and_filter_channel_updates(data_path)
    print(f"Loaded {len(df)} channel updates")
    
    print(f"\nComputing effective fees for {amount_sats:,} sats...")
    df = compute_effective_fee_and_lags(df, amount_sats)
    
    channel_id, channel_df = select_best_channel(df)
    
    results = train_and_forecast(channel_df, time_budget)
    
    print_results(results, channel_id)
    
    return results

