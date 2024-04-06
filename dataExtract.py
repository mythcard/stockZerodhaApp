import argparse
from kiteconnect import KiteConnect
import datetime

# Define the mapping of instrument tokens to symbols
token_symbol_map = {
    "408065": "INFY",
    "738561": "RELIANCE",
    "884737": "TATAMOTORS",
    "341249": "HDFCBANK",
    "779521": "SBIN",
    "4752385": "LTTS"
}

# Function to fetch data
def fetch_data(kite, start_date, end_date, instrument_token, interval):
    return kite.historical_data(instrument_token, start_date, end_date, interval, False, True)


def format_date(date_obj):
    # Convert the datetime object to a string in the desired format
    return date_obj.strftime('%Y-%m-%d %H:%M:%S')


def write_to_psv(data, run_date, instrument_token, filename="data.psv"):
    with open(filename, "w") as f:
        # Write the header
        f.write("datetime|open|high|low|close|volume\n")
        for item in data:
            f.write(
                f"{format_date(item['date'])}|{item['open']}|{item['high']}|{item['low']}|{item['close']}|{item['volume']}\n")


def main(api_key, access_token, api_secret, start_date, end_date, instrument_token_list, future_step, interval,
         time_step):
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(access_token, api_secret)
    kite.set_access_token(data["access_token"])

    # Convert instrument_token_list string into a list of tokens
    instrument_tokens = instrument_token_list.split("-")

    # Get current run date in YYYYMMDD format
    run_date = datetime.datetime.now().strftime("%Y%m%d")


    for instrument_token in instrument_tokens:
        # Fetch data
        histData = fetch_data(kite, start_date, end_date, instrument_token, interval)

        # Get the symbol from the token_symbol_map
        symbol = token_symbol_map.get(instrument_token, "Unknown")

        # Filename based on runDate, instrument ID, and "symbol" (assuming instrument_token as symbol here)
        filename = f"{run_date}-{instrument_token}-{symbol}.psv"

        # Format and write data to a PSV file
        write_to_psv(histData, run_date, instrument_token, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Kite Connect API key")
    parser.add_argument("--access_token", type=str, required=True, help="Kite Connect Access Token")
    parser.add_argument("--api_secret", type=str, required=True, help="Kite API secret")
    parser.add_argument("--start_date", type=str, required=True,
                        help="Start date for historical data in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, required=True, help="End date for historical data in YYYY-MM-DD format")
    parser.add_argument("--instrument_token_list", type=str, required=True,
                        help="List of instrument tokens for the stock, separated by '-'")
    parser.add_argument("--future_step", type=int, choices=[5, 10, 30, 60], required=True,
                        help="Prediction minute (5, 10, 30, 60)")
    parser.add_argument("--interval", type=str, default="minute", help="Interval for historical data")
    parser.add_argument("--time_step", type=int, default=100, help="Number of time steps for the LSTM model")

    args = parser.parse_args()

    main(args.api_key, args.access_token, args.api_secret, args.start_date, args.end_date, args.instrument_token_list,
         args.future_step, args.interval, args.time_step)
