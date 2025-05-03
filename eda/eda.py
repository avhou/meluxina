import argparse
import sqlite3
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_count_and_percentage(db: str, source: str) -> Tuple[int, float]:
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            f"select count(*) from articles where source = ?", (source,)
        ).fetchone()[0]
        total_count = conn.execute(f"select count(*) from articles").fetchone()[0]
        return count, count / total_count


def get_date_range(db: str, source: str) -> Tuple[str, str]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            f"select min(timestamp), max(timestamp) from articles where source = ?",
            (source,),
        ).fetchone()


def generate_time_distribution(db: str, source: str, image_output_dir: str):
    with sqlite3.connect(db) as conn:
        query = """ SELECT timestamp FROM articles order by timestamp asc"""

        df = pd.read_sql_query(query, conn)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["month"] = df["timestamp"].dt.to_period("M")
        monthly_counts = df.groupby("month").size().sort_index()

        # Convert PeriodIndex to datetime for plotting
        monthly_counts.index = monthly_counts.index.to_timestamp()

        plt.figure(figsize=(12, 6))
        monthly_counts.plot(kind="bar", color="skyblue")
        plt.title(f"Article Distribution Over Time ({source})")
        plt.xlabel("Month")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.xticks(rotation=45)

        plt.savefig(os.path.join(image_output_dir, f"{source}_time_distribution.png"))
        plt.close()


def eda(non_threaded_db: str, threaded_db: str, image_output_dir: str, source: str):
    print(
        f"performing analysis in {non_threaded_db} / {threaded_db} for source {source}"
    )

    count_non_threaded, percentage_non_threaded = get_count_and_percentage(
        non_threaded_db, source
    )
    count_threaded, percentage_threaded = get_count_and_percentage(threaded_db, source)
    print(
        f"non threaded count {count_non_threaded}, percentage {percentage_non_threaded * 100:.2f}"
    )
    print(
        f"threaded count {count_threaded}, percentage {percentage_threaded * 100:.2f}"
    )

    daterange_min_non_threaded, daterange_max_non_threaded = get_date_range(
        non_threaded_db, source
    )
    daterange_min_threaded, daterange_max_threaded = get_date_range(threaded_db, source)
    print(
        f"non threaded daterange_min {daterange_min_non_threaded}, daterange_max {daterange_max_non_threaded}"
    )
    print(
        f"threaded daterange_min {daterange_min_threaded}, daterange_max {daterange_max_threaded}"
    )

    generate_time_distribution(non_threaded_db, source, image_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA")

    parser.add_argument(
        "--non-threaded-db",
        type=str,
        required=True,
        help="Path to the non-threaded database file",
    )
    parser.add_argument(
        "--threaded-db",
        type=str,
        required=True,
        help="Path to the threaded database file",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source to consider",
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        required=True,
        help="Path to the image output directory",
    )
    args = parser.parse_args()

    eda(args.non_threaded_db, args.threaded_db, args.image_output_dir, args.source)

    print("done")
