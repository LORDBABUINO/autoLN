import argparse

from ln_flaml import run_forecast


def main():
    parser = argparse.ArgumentParser(
        description="Lightning Network Fee Forecasting with FLAML"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ln_node_data.json",
        help="Path to LN node data JSON file (default: data/ln_node_data.json)",
    )
    parser.add_argument(
        "--amount_sats",
        type=int,
        default=100_000,
        help="Payment amount in satoshis for fee calculation (default: 100000)",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=30,
        help="FLAML training time budget in seconds (default: 30)",
    )

    args = parser.parse_args()

    print("="*60)
    print("LN Fee Forecasting with FLAML")
    print("="*60)
    print(f"Data path: {args.data}")
    print(f"Payment amount: {args.amount_sats:,} sats")
    print(f"Time budget: {args.time_budget}s")
    print("="*60)

    try:
        run_forecast(
            data_path=args.data,
            amount_sats=args.amount_sats,
            time_budget=args.time_budget,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
