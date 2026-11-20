"""Analyze flight delays and save cleaned data and summary tables.

Reads data/raw/14_flight_delays.csv and writes results to output/.
Requires pandas and numpy.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path("data/raw/14_flight_delays.csv")
OUTPUT_DIR = Path("output")


def percent_yes(values):
    """Return the percentage of affirmative values in a Series."""
    return values.astype(str).str.strip().str.lower().eq("yes").mean() * 100


def main():
    # Load the flights and standardize fields used in calculations.
    df = pd.read_csv(DATA_FILE)
    duplicates = df["Flight_ID"].duplicated().sum()
    df = df.drop_duplicates(subset="Flight_ID")

    for column in ("Departure_Delay_Mins", "Arrival_Delay_Mins"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Arrival_Delay_Mins"] = df["Arrival_Delay_Mins"].fillna(
        df["Arrival_Delay_Mins"].median()
    )
    df["Flight_Date"] = pd.to_datetime(df["Flight_Date"], errors="coerce")
    df["Month"] = df["Flight_Date"].dt.month
    df["Day_of_Week"] = df["Flight_Date"].dt.day_name()

    print("Dataset shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Duplicate Flight_IDs removed:", duplicates)
    print("Median arrival delay used:", df["Arrival_Delay_Mins"].median())

    # Summarize punctuality and airport or route performance.
    total = len(df)
    on_time = df["Arrival_Delay_Mins"].le(15).sum()
    on_time_percentage = on_time / total * 100 if total else 0

    print(
        f"Flights: {total} | Mean departure delay: "
        f"{df['Departure_Delay_Mins'].mean():.2f} min"
    )
    print(
        f"On-time arrivals (15 minutes or less): "
        f"{on_time} ({on_time_percentage:.2f}%)"
    )

    airport_ranking = (
        df.groupby("Origin")["Departure_Delay_Mins"]
        .mean()
        .sort_values(ascending=False)
    )

    df["Route"] = (
        df["Origin"].astype(str) + " to " + df["Destination"].astype(str)
    )
    worst_routes = (
        df.groupby("Route")["Arrival_Delay_Mins"].mean().nlargest(10)
    )

    # Convert departure times into broad periods for delay comparisons.
    hour = pd.to_datetime(
        df["Departure_Time"].astype(str),
        format="mixed",
        errors="coerce",
    ).dt.hour

    df["Departure_Hour"] = hour
    df["Time_of_Day"] = np.select(
        [hour.between(5, 11), hour.between(12, 16), hour.between(17, 20)],
        ["Morning", "Afternoon", "Evening"],
        default="Night",
    )

    time_analysis = (
        df.groupby("Time_of_Day")["Departure_Delay_Mins"]
        .mean()
        .sort_values(ascending=False)
    )

    # Report optional causes and flight-status rates when those fields exist.
    if "Delay_Cause" in df:
        print("Delay causes:\n", df["Delay_Cause"].value_counts())

    for column, label in (
        ("Cancelled", "Cancellation"),
        ("Diverted", "Diversion"),
    ):
        if column in df:
            print(
                f"{label} percentage by airline:\n",
                df.groupby("Airline")[column].apply(percent_yes),
            )

    # Measure the relationship, spread, and extreme values of departure delays.
    correlation = df[
        ["Departure_Delay_Mins", "Arrival_Delay_Mins"]
    ].corr().iloc[0, 1]
    print("Departure/arrival delay correlation:", round(correlation, 2))

    delays = df["Departure_Delay_Mins"].dropna().to_numpy()

    if delays.size:
        q1, q3 = np.percentile(delays, [25, 75])
        lower = q1 - 1.5 * (q3 - q1)
        upper = q3 + 1.5 * (q3 - q1)

        outliers = df[
            df["Departure_Delay_Mins"].lt(lower)
            | df["Departure_Delay_Mins"].gt(upper)
        ]

        print(
            "Delay median / standard deviation / 90th percentile:",
            np.round(
                [
                    np.median(delays),
                    np.std(delays),
                    np.percentile(delays, 90),
                ],
                2,
            ),
        )
        print(
            f"IQR outlier limits: {lower:.2f} to {upper:.2f} "
            f"| Outliers: {len(outliers)}"
        )
        print("Largest-delay outlier records:\n", outliers.head(10))
    else:
        outliers = df.iloc[0:0]
        print("No valid departure-delay values available for statistics.")

    # Label delays and retain the cleaned dataset plus key summary tables.
    df["Delay_Severity"] = np.select(
        [
            df["Departure_Delay_Mins"].le(15),
            df["Departure_Delay_Mins"].between(16, 60),
            df["Departure_Delay_Mins"].between(61, 180),
            df["Departure_Delay_Mins"].gt(180),
        ],
        ["On Time", "Minor Delay", "Major Delay", "Severe Delay"],
        default="Unknown",
    )

    print("Delay severity counts:\n", df["Delay_Severity"].value_counts())
    print(
        "Recommendations: review airport turnaround times, high-delay routes "
        "and peak periods;"
    )
    print(
        "monitor delay causes and give severe-delay flights operational attention."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_DIR / "cleaned_flight_data.csv", index=False)

    airport_ranking.rename_axis("Origin").reset_index(
        name="Departure_Delay_Mins"
    ).to_csv(OUTPUT_DIR / "airport_delay_ranking.csv", index=False)

    worst_routes.rename_axis("Route").reset_index(
        name="Arrival_Delay_Mins"
    ).to_csv(OUTPUT_DIR / "worst_10_routes.csv", index=False)

    time_analysis.rename_axis("Time_of_Day").reset_index(
        name="Departure_Delay_Mins"
    ).to_csv(OUTPUT_DIR / "time_of_day_analysis.csv", index=False)

    print(f"Analysis complete. Results saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()