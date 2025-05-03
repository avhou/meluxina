import argparse
import sqlite3
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math
import re
import statistics
import numpy as np


def exec(db: str, query: str):
    with sqlite3.connect(db) as conn:
        return conn.execute(query).fetchone()[0]


def get_count_and_percentage(db: str, source: str) -> Tuple[int, float]:
    with sqlite3.connect(db) as conn:
        count = conn.execute(f"select count(*) from articles where source = ?", (source,)).fetchone()[0]
        total_count = conn.execute(f"select count(*) from articles").fetchone()[0]
        return count, count / total_count


def get_date_range(db: str, source: str) -> Tuple[str, str]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            f"select min(timestamp), max(timestamp) from articles where source = ?",
            (source,),
        ).fetchone()


def do_generate_time_distribution_sns(db: str, source: str, output_file: str, query: str, title: str, y_lim: int | None = None) -> int:
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(query, conn)

        # Clean up the timestamp and convert it to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"].str.replace(r"\s[+\-]\d{2}$", ""), errors="coerce")

        # Create a 'month' column to group by
        df["month"] = df["timestamp"].dt.to_period("M")
        monthly_counts = df.groupby("month").size().sort_index()

        # Create an index of all months from 2022-01 to 2024-11
        all_months = pd.date_range(start="2022-01-01", end="2024-11-01", freq="MS").to_period("M")

        # Reindex monthly_counts to include all months, filling missing values with 0
        monthly_counts = monthly_counts.reindex(all_months, fill_value=0)

        # Convert the PeriodIndex to datetime for plotting
        monthly_counts.index = monthly_counts.index.to_timestamp()

        # Create a DataFrame for Seaborn
        monthly_df = pd.DataFrame({"month": monthly_counts.index, "count": monthly_counts.values})

        # Plotting with Seaborn
        plt.figure(figsize=(12, 6))

        # Use Seaborn's barplot
        sns.barplot(x="month", y="count", data=monthly_df, color="skyblue")

        # Set title and labels
        plt.title(title, fontsize=16)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Number of Articles", fontsize=12)

        # Format x-axis labels to show only the 'YYYY-MM' format
        plt.xticks(rotation=45, fontsize=10)
        if y_lim is not None:
            plt.ylim(top=y_lim)
        else:
            plt.ylim(top=math.ceil(monthly_counts.values.max() / 10.0) * 10)

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(output_file)
        plt.close()

        return math.ceil(monthly_counts.values.max() / 10.0) * 10


def generate_lang_distribution(db: str, source: str, output_file: str):
    total = exec(db, f"""select count(*) from articles where source = '{source}'""")
    with sqlite3.connect(db) as conn, open(output_file, "w") as f:
        f.write("| language | count  | % |\n")
        f.write("|--------: | -----: | ------:|\n")

        for lang, count in conn.execute(
            f"""select detected_language, count(*) from articles where source = '{source}' group by detected_language order by 1 asc"""
        ):
            print(f" lang {lang} has count {count}")
            f.write(f"| {lang} | {count} | {count / total:.2f} |\n")


def get_word_count(s: str) -> int:
    cleaned = re.sub(r"\s+", " ", s).strip()
    return len(cleaned.split()) if cleaned else 0


def generate_word_count_distribution(db: str, source: str, output_file: str):
    with sqlite3.connect(db) as conn, open(output_file, "w") as f:
        f.write("| word count metric | value  | \n")
        f.write("|--------: | -----: |\n")

        texts = []
        for (text,) in conn.execute(f"select text from articles where source = '{source}'"):
            texts.append(text)

        word_counts = [get_word_count(text) for text in texts]
        # Basic statistics
        minimum = min(word_counts)
        maximum = max(word_counts)
        mean = statistics.mean(word_counts)
        median = statistics.median(word_counts)

        # Percentiles using numpy
        percentile_25 = np.percentile(word_counts, 25)
        percentile_50 = np.percentile(word_counts, 50)  # same as median
        percentile_75 = np.percentile(word_counts, 75)
        percentile_100 = np.percentile(word_counts, 100)

        f.write(f"| min | {minimum} |\n")
        f.write(f"| max | {maximum} |\n")
        f.write(f"| mean | {mean:.2f} |\n")
        f.write(f"| median | {median:.2f} |\n")
        f.write(f"| Q1 | {percentile_25:.2f} |\n")
        f.write(f"| Q2 | {percentile_50:.2f} |\n")
        f.write(f"| Q3 | {percentile_75:.2f} |\n")
        f.write(f"| Q4 | {percentile_100:.2f} |\n")


def generate_time_distribution_sns(db: str, source: str, image_output_dir: str):
    max_count = do_generate_time_distribution_sns(
        db,
        source,
        os.path.join(image_output_dir, f"{source}_time_distribution.png"),
        f"""SELECT timestamp FROM articles WHERE source = '{source}' ORDER BY timestamp ASC""",
        f"Article Count Over Time ({source})",
        None,
    )
    do_generate_time_distribution_sns(
        db,
        source,
        os.path.join(image_output_dir, f"{source}_disinformation_time_distribution.png"),
        f"""SELECT timestamp FROM articles WHERE source = '{source}' and disinformation = 'y' ORDER BY timestamp ASC""",
        f"Disinformation Article Count Over Time ({source})",
        max_count,
    )


def eda(non_threaded_db: str, threaded_db: str, output_dir: str, source: str):
    print(f"performing analysis in {non_threaded_db} / {threaded_db} for source {source}")

    count_non_threaded, percentage_non_threaded = get_count_and_percentage(non_threaded_db, source)
    count_threaded, percentage_threaded = get_count_and_percentage(threaded_db, source)
    print(f"non threaded count {count_non_threaded}, percentage {percentage_non_threaded * 100:.2f}")
    print(f"threaded count {count_threaded}, percentage {percentage_threaded * 100:.2f}")

    daterange_min_non_threaded, daterange_max_non_threaded = get_date_range(non_threaded_db, source)
    daterange_min_threaded, daterange_max_threaded = get_date_range(threaded_db, source)
    print(f"non threaded daterange_min {daterange_min_non_threaded}, daterange_max {daterange_max_non_threaded}")
    print(f"threaded daterange_min {daterange_min_threaded}, daterange_max {daterange_max_threaded}")

    generate_time_distribution_sns(non_threaded_db, source, os.path.join(output_dir, "images"))

    generate_lang_distribution(non_threaded_db, source, os.path.join(os.path.join(output_dir, "tables"), f"{source}_lang_distribution.md"))
    generate_word_count_distribution(non_threaded_db, source, os.path.join(os.path.join(output_dir, "tables"), f"{source}_word_count_distribution.md"))


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
        "--output-dir",
        type=str,
        required=True,
        help="Path to the image output directory",
    )
    args = parser.parse_args()

    eda(args.non_threaded_db, args.threaded_db, args.output_dir, args.source)

    print("done")
