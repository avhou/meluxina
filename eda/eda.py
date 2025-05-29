import argparse
import sqlite3
from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math
import re
import statistics
import numpy as np
import json
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import spacy
from spacy.cli import download
from pathlib import Path
from nltk.util import ngrams
from collections import Counter
from itertools import islice, chain


dutch_stopwords = set(
    [
        "de",
        "dus",
        "der",
        "en",
        "een",
        "het",
        "van",
        "aan",
        "in",
        "met",
        "te",
        "voor",
        "als",
        "op",
        "is",
        "was",
        "heeft",
        "waar",
        "door",
        "niet",
        "er",
        "maar",
        "ook",
        "ze",
        "die",
        "dat",
        "om",
        "zijn",
        "dit",
        "al",
        "dan",
        "bij",
        "hij",
        "zij",
        "naar",
        "nu",
        "zich",
        "nog",
        "wel",
        "hun",
        "ons",
        "mijn",
        "zo",
        "heel",
        "ik",
        "tot",
        "omdat",
        "daar",
        "uit",
        "hier",
        "wat",
    ]
)

french_stopwords = set(
    [
        "le",
        "la",
        "les",
        "de",
        "des",
        "en",
        "et",
        "à",
        "un",
        "une",
        "ce",
        "que",
        "qui",
        "pour",
        "dans",
        "avec",
        "sur",
        "par",
        "ne",
        "pas",
        "je",
        "selon",
        "il",
        "se",
        "du",
        "est",
    ]
)

keywords = list(
    set(
        [
            "migrant",
            "immigrant",
            "emigrant",
            "migratie",
            "immigratie",
            "vluchteling",
            "oorlogsvluchteling",
            "ontheemde",
            "vluchtende bevolking",
            "verspreide bevolking",
            "herplaatste bevolking",
            "asielzoeker",
            "diaspora",
            "buitenlander",
            "expat",
            "migrant",
            "immigrant",
            "émigrant",
            "migration",
            "immigration",
            "réfugié",
            "réfugiés de guerre",
            "personnes déplacées",
            "population fuyante",
            "population dispersée",
            "personne relocalisée",
            "demandeur d'asile",
            "diaspora",
            "étranger",
            "expatrié",
            "migrant",
            "immigrant",
            "emigrant",
            "migration",
            "immigration",
            "refugee",
            "war refugees",
            "displaced people",
            "fleeing population",
            "dispersed population",
            "relocated population",
            "asylum seeker",
            "diaspora",
            "expatriate",
            "expat",
            "oekraïne",
            "rusland",
            "oekraine",
            "russie",
            "ukraine",
            "russia",
        ]
    )
)


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
        df["timestamp"] = pd.to_datetime(df["timestamp"].str.replace(r"\s[+\-]\d{2}$", ""), errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

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


def get_stopword_count(s: str) -> int:
    all_stopwords = set(dutch_stopwords)
    all_stopwords.update(french_stopwords)
    all_stopwords.update(STOPWORDS)
    cleaned = re.sub(r"\s+", " ", s).strip()
    if cleaned:
        return sum([1 if word in all_stopwords else 0 for word in cleaned.split()])
    return 0


def get_token_count(s: str) -> List[int]:
    cleaned = re.sub(r"\s+", " ", s).strip()
    if cleaned:
        return [
            len(word) for word in cleaned.split() if word != "." and word != "?" and word != "!" and word != "," and word != "-" and word != "_" and word != "–"
        ]
    return []


def generate_word_count_distribution(db: str, source: str, output_file: str):
    with sqlite3.connect(db) as conn, open(output_file, "w") as f:
        f.write("| Word Count Metric | value  | \n")
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


def analyze_metadata_distribution_web(db: str, source: str, output_dir: str):
    with sqlite3.connect(db) as conn:
        # Read metadata from the article table
        df = pd.read_sql_query(f"SELECT metadata, disinformation FROM articles where source = '{source}' ", conn)

    df_parsed = df["metadata"].apply(json.loads).apply(pd.Series)
    df_parsed["disinformation"] = df["disinformation"].map({"y": 1, "n": 0}).fillna(0).astype(int)

    # Group by source type
    grouped = df_parsed.groupby("political_party").agg(count=("political_party", "size"), disinformation_count=("disinformation", "sum"))
    grouped["count"] = grouped["count"].astype(int)
    grouped["disinformation_count"] = grouped["disinformation_count"].astype(int)
    grouped["percentage"] = (grouped["count"] / grouped["count"].sum() * 100).round(2)
    grouped["disinfo_percentage"] = (grouped["disinformation_count"] / grouped["count"] * 100).round(2)

    # Write total percentage table
    with open(os.path.join(output_dir, f"{source}_metadata_count.md"), "w") as f:
        f.write("| Source | Count | Percentage |\n")
        f.write("|----------------:|------:|-----------:|\n")
        for political_party, row in grouped.iterrows():
            label = "News outlet" if political_party == 0 else "Political party"
            f.write(f"| {label} | {int(row['count'])} | {row['percentage']:.2f} % |\n")

    with open(os.path.join(output_dir, f"{source}_metadata_disinfo.md"), "w") as f:
        f.write("| Source | Total count | Disinformation count | Disinformation percentage |\n")
        f.write("|----------------:|------:|----------------:|---------------------:|\n")
        for idx, row in grouped.iterrows():
            label = "News outlet" if idx == 0 else "Political party"
            f.write(f"| {label} | {int(row['count'])} | {int(row['disinformation_count'])} | {row['disinfo_percentage']:.2f} % |\n")

    # Process political parties per host with disinformation details
    political_parties = df_parsed[df_parsed["political_party"] == 1]
    political_parties_per_host = political_parties.groupby("host").agg(count=("host", "size"), disinformation_count=("disinformation", "sum"))
    political_parties_per_host["disinfo_percentage"] = (political_parties_per_host["disinformation_count"] / political_parties_per_host["count"] * 100).round(2)
    # Sort by count in descending order
    political_parties_per_host = political_parties_per_host.sort_values(by="count", ascending=False)

    with open(os.path.join(output_dir, f"{source}_political_party.md"), "w") as f:
        f.write("| Political party | Count | Disinformation count | Disinformation percentage |\n")
        f.write("|-----------------:|------:|---------------------:|--------------------------:|\n")
        for host, row in political_parties_per_host.iterrows():
            f.write(f"| {host} | {int(row['count'])} | {int(row['disinformation_count'])} | {row['disinfo_percentage']:.2f} % |\n")

    # Process news outlets per host with disinformation details
    news_outlets = df_parsed[df_parsed["political_party"] == 0]
    news_outlets_per_host = news_outlets.groupby("host").agg(count=("host", "size"), disinformation_count=("disinformation", "sum"))
    news_outlets_per_host["disinfo_percentage"] = (news_outlets_per_host["disinformation_count"] / news_outlets_per_host["count"] * 100).round(2)
    news_outlets_per_host = news_outlets_per_host.sort_values(by="count", ascending=False)

    with open(os.path.join(output_dir, f"{source}_news_outlets.md"), "w") as f:
        f.write("| News outlet | Count | Disinformation count | Disinformation % |\n")
        f.write("|-------------:|------:|---------------------:|--------------------------:|\n")
        for host, row in news_outlets_per_host.iterrows():
            f.write(f"| {host} | {int(row['count'])} | {int(row['disinformation_count'])} | {row['disinfo_percentage']:.2f} % |\n")


def analyze_metadata_distribution_reddit(non_threaded_db: str, threaded_db: str, source: str, output_dir: str):
    with sqlite3.connect(non_threaded_db) as conn:
        # Read metadata from the article table
        df = pd.read_sql_query(f"SELECT metadata, disinformation FROM articles where source = '{source}' ", conn)

    df_parsed = df["metadata"].apply(json.loads).apply(pd.Series)
    df_parsed["disinformation"] = df["disinformation"].map({"y": 1, "n": 0}).fillna(0).astype(int)

    # Group by source type
    grouped = df_parsed.groupby("subreddit").agg(count=("subreddit", "size"), disinformation_count=("disinformation", "sum"))
    grouped["count"] = grouped["count"].astype(int)
    grouped["disinformation_count"] = grouped["disinformation_count"].astype(int)
    grouped["percentage"] = (grouped["count"] / grouped["count"].sum() * 100).round(2)
    grouped["disinfo_percentage"] = (grouped["disinformation_count"] / grouped["count"] * 100).round(2)

    # Write total percentage table
    with open(os.path.join(output_dir, f"{source}_non_threaded_metadata_count.md"), "w") as f:
        f.write("| Source | Count | % |\n")
        f.write("|----------------:|------:|-----------:|\n")
        for subreddit, row in grouped.iterrows():
            f.write(f"| {subreddit} | {int(row['count'])} | {row['percentage']:.2f} % |\n")

    with open(os.path.join(output_dir, f"{source}_non_threaded_metadata_disinfo.md"), "w") as f:
        f.write("| Source | Total count | Disinformation count | Disinformation % |\n")
        f.write("|----------------:|------:|----------------:|---------------------:|\n")
        for subreddit, row in grouped.iterrows():
            f.write(f"| {subreddit} | {int(row['count'])} | {int(row['disinformation_count'])} | {row['disinfo_percentage']:.2f} % |\n")

    with sqlite3.connect(threaded_db) as conn:
        # Read metadata from the article table
        df = pd.read_sql_query(f"SELECT metadata, disinformation FROM articles where source = '{source}' ", conn)

    df_parsed = df["metadata"].apply(json.loads).apply(pd.Series)
    df_parsed["disinformation"] = df["disinformation"].map({"y": 1, "n": 0}).fillna(0).astype(int)

    # Group by source type
    grouped = df_parsed.groupby("subreddit").agg(count=("subreddit", "size"), disinformation_count=("disinformation", "sum"))
    grouped["count"] = grouped["count"].astype(int)
    grouped["disinformation_count"] = grouped["disinformation_count"].astype(int)
    grouped["percentage"] = (grouped["count"] / grouped["count"].sum() * 100).round(2)
    grouped["disinfo_percentage"] = (grouped["disinformation_count"] / grouped["count"] * 100).round(2)

    # Write total percentage table
    with open(os.path.join(output_dir, f"{source}_threaded_metadata_count.md"), "w") as f:
        f.write("| Source | Count | % |\n")
        f.write("|----------------:|------:|-----------:|\n")
        for subreddit, row in grouped.iterrows():
            f.write(f"| {subreddit} | {int(row['count'])} | {row['percentage']:.2f} % |\n")

    with open(os.path.join(output_dir, f"{source}_threaded_metadata_disinfo.md"), "w") as f:
        f.write("| Source | Total count | Disinformation count | Disinformation % |\n")
        f.write("|----------------:|------:|----------------:|---------------------:|\n")
        for subreddit, row in grouped.iterrows():
            f.write(f"| {subreddit} | {int(row['count'])} | {int(row['disinformation_count'])} | {row['disinfo_percentage']:.2f} % |\n")


def keyword_distribution(db: str, source: str, output_dir: str):
    escaped_keywords = [re.escape(k) for k in keywords]
    pattern = r"(?i)\b(?:" + "|".join(escaped_keywords) + r")"
    frequencies = Counter()
    with sqlite3.connect(db) as conn:
        for (text,) in conn.execute(f"select text from articles where source = '{source}'"):
            matches = re.findall(pattern, text)
            # Normalize to lowercase for consistent counting
            frequencies.update(match.lower() for match in matches)

    with open(os.path.join(output_dir, f"{source}_keyword_distribution.md"), "w") as f:
        f.write("| Keyword | Count |\n")
        f.write("| :----- | -----: |\n")
        for keyword, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {keyword} | {count} |\n")


def generate_word_cloud(db: str, source: str, output_dir: str):
    texts = []
    stopwords = STOPWORDS.union(dutch_stopwords).union(french_stopwords)
    with sqlite3.connect(db) as conn:
        for (text,) in conn.execute(f"select text from articles where source = '{source}'"):
            texts.append(re.sub(r"\s+", " ", text).strip())
    combined_text = " ".join(texts).lower()
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords).generate(combined_text)
    wordcloud.to_file(os.path.join(output_dir, f"{source}_word_cloud.png"))
    texts = []
    with sqlite3.connect(db) as conn:
        for (text,) in conn.execute(f"select translated_text from articles where source = '{source}'"):
            texts.append(re.sub(r"\s+", " ", text).strip())
    combined_text = " ".join(texts).lower()
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords).generate(combined_text)
    wordcloud.to_file(os.path.join(output_dir, f"{source}_translated_word_cloud.png"))


def generate_token_distribution(db: str, source: str, output_file: str):
    with sqlite3.connect(db) as conn, open(output_file, "w") as f:
        f.write("| Word length metric | value  | \n")
        f.write("|:-------- | -----: |\n")

        texts = []
        for (text,) in conn.execute(f"select translated_text from articles where source = '{source}'"):
            texts.append(text)

        word_lengths = [token_length for text in texts for token_length in get_token_count(text)]
        # Basic statistics
        minimum = min(word_lengths)
        maximum = max(word_lengths)
        mean = statistics.mean(word_lengths)
        median = statistics.median(word_lengths)

        f.write(f"| min | {minimum} |\n")
        f.write(f"| max | {maximum} |\n")
        f.write(f"| mean | {mean:.2f} |\n")
        f.write(f"| median | {median:.2f} |\n")


def generate_stopword_ratio(db: str, source: str, output_dir: str):
    with sqlite3.connect(db) as conn:
        texts = []
        translated_texts = []
        for text, translated_text in conn.execute(f"select text, translated_text from articles where source = '{source}'"):
            texts.append(text)
            translated_texts.append(translated_text)

        word_counts_texts = [get_word_count(text) for text in texts]
        stopword_counts_texts = [get_stopword_count(text) for text in texts]
        stopword_ratios_texts = []
        for word_count, stopword_count in zip(word_counts_texts, stopword_counts_texts):
            stopword_ratios_texts.append(stopword_count / word_count if word_count > 0 else 0)

        word_counts_translated_texts = [get_word_count(text) for text in translated_texts]
        stopword_counts_translated_texts = [get_stopword_count(text) for text in translated_texts]
        stopword_ratios_translated_texts = []
        for word_count, stopword_count in zip(word_counts_translated_texts, stopword_counts_translated_texts):
            stopword_ratios_translated_texts.append(stopword_count / word_count if word_count > 0 else 0)

        with open(os.path.join(output_dir, f"{source}_stopword_ratio_texts.md"), "w") as f:
            f.write("| Stopword ratio metric | Value  | \n")
            f.write("|:-------- | -----: |\n")

            # Basic statistics
            minimum = min(stopword_ratios_texts)
            maximum = max(stopword_ratios_texts)
            mean = statistics.mean(stopword_ratios_texts)
            median = statistics.median(stopword_ratios_texts)

            f.write(f"| min | {minimum:.2f} |\n")
            f.write(f"| max | {maximum:.2f} |\n")
            f.write(f"| mean | {mean:.2f} |\n")
            f.write(f"| median | {median:.2f} |\n")

        with open(os.path.join(output_dir, f"{source}_stopword_ratio_translated_texts.md"), "w") as f:
            f.write("| Stopword ratio metric | Value  | \n")
            f.write("|:-------- | -----: |\n")

            # Basic statistics
            minimum = min(stopword_ratios_translated_texts)
            maximum = max(stopword_ratios_translated_texts)
            mean = statistics.mean(stopword_ratios_translated_texts)
            median = statistics.median(stopword_ratios_translated_texts)

            f.write(f"| min | {minimum:.2f} |\n")
            f.write(f"| max | {maximum:.2f} |\n")
            f.write(f"| mean | {mean:.2f} |\n")
            f.write(f"| median | {median:.2f} |\n")


def generate_pie_charts(output_file: str, counts: List[int], labels: List[str]):
    explode = [0.1] + [0] * (len(counts) - 1)

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct="%1.2f%%", startangle=90, shadow=True, explode=explode)
    plt.axis("equal")  # Ensures the pie is a circle
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file)
    plt.close()


def load_spacy_model(name):
    try:
        return spacy.load(name)
    except OSError:
        print(f"Model {name} not found. Downloading...")
        download(name)
        return spacy.load(name)


def generate_ngrams(db: str, output_dir: str, source: str):
    print(f"generating ngrams for {db} and source {source}")
    english_stopwords = set(STOPWORDS)
    english_stopwords.update(["of", "the", "a", "to", "on", "tankts", "pa", "schepenen", "𝘴𝘤𝘩𝘦𝘱𝘦𝘯𝘦𝘯"])
    spacy_models = {"en": load_spacy_model("en_core_web_sm"), "fr": load_spacy_model("fr_core_news_sm"), "nl": load_spacy_model("nl_core_news_sm")}
    custom_stopwords = {"en": english_stopwords, "fr": set(french_stopwords), "nl": set(dutch_stopwords)}

    token_lists = []
    with sqlite3.connect(db) as conn:
        for (text,) in conn.execute(f"select translated_text from articles where source = ?", (source,)).fetchall():
            language = "en"
            nlp = spacy_models[language]
            doc = nlp(text)

            # Tokenize, lowercase, filter punctuation/stopwords
            tokens = [token.lemma_.lower() for token in doc if not token.is_punct and token.lemma_.lower() not in custom_stopwords[language]]
            token_lists.append(tokens)

    def flatten_ngrams(token_lists, n):
        return list(chain.from_iterable(ngrams(tokens, n) for tokens in token_lists if len(tokens) >= n))

    bigram_counts = Counter(flatten_ngrams(token_lists, 2))
    trigram_counts = Counter(flatten_ngrams(token_lists, 3))

    with open(os.path.join(output_dir, f"2_gram_{source}.md"), "w") as f:
        f.write("| bigram | count | \n")
        f.write("| ----- | ----: | \n")
        for ngram, count in bigram_counts.most_common(10):
            f.write(f"| {' '.join(ngram)} | {count} |\n")

    with open(os.path.join(output_dir, f"3_gram_{source}.md"), "w") as f:
        f.write("| trigram | count | \n")
        f.write("| ----- | ----: | \n")
        for ngram, count in trigram_counts.most_common(10):
            f.write(f"| {' '.join(ngram)} | {count} |\n")


def eda(non_threaded_db: str, threaded_db: str, output_dir: str, source: str):
    print(f"performing analysis in {non_threaded_db} / {threaded_db} for source {source}")

    images = os.path.join(output_dir, "images")
    tables = os.path.join(output_dir, "tables")

    count_non_threaded, percentage_non_threaded = get_count_and_percentage(non_threaded_db, source)
    count_threaded, percentage_threaded = get_count_and_percentage(threaded_db, source)
    print(f"non threaded count {count_non_threaded}, percentage {percentage_non_threaded * 100:.2f}")
    print(f"threaded count {count_threaded}, percentage {percentage_threaded * 100:.2f}")

    daterange_min_non_threaded, daterange_max_non_threaded = get_date_range(non_threaded_db, source)
    daterange_min_threaded, daterange_max_threaded = get_date_range(threaded_db, source)
    print(f"non threaded daterange_min {daterange_min_non_threaded}, daterange_max {daterange_max_non_threaded}")
    print(f"threaded daterange_min {daterange_min_threaded}, daterange_max {daterange_max_threaded}")

    generate_time_distribution_sns(non_threaded_db, source, images)

    generate_lang_distribution(non_threaded_db, source, os.path.join(tables, f"{source}_lang_distribution.md"))
    generate_word_count_distribution(non_threaded_db, source, os.path.join(tables, f"{source}_word_count_distribution.md"))
    if source == "web":
        analyze_metadata_distribution_web(non_threaded_db, source, tables)
    keyword_distribution(non_threaded_db, source, tables)
    generate_word_cloud(non_threaded_db, source, images)
    generate_token_distribution(non_threaded_db, source, os.path.join(tables, f"{source}_token_distribution.md"))
    generate_stopword_ratio(non_threaded_db, source, tables)
    if source == "reddit":
        analyze_metadata_distribution_reddit(non_threaded_db, threaded_db, source, tables)
    generate_pie_charts(
        os.path.join(images, "web_proportion_articles.png"),
        [252, 88, 55],
        ["Web", "TikTok", "Reddit"],
    )
    generate_pie_charts(
        os.path.join(images, "web_proportion_threads.png"),
        [252, 88, 1519],
        ["Web", "TikTok", "Reddit"],
    )
    generate_pie_charts(
        os.path.join(images, "tiktok_proportion_articles.png"),
        [88, 252, 55],
        ["TikTok", "Web", "Reddit"],
    )
    generate_pie_charts(
        os.path.join(images, "tiktok_proportion_threads.png"),
        [88, 252, 1519],
        ["TikTok", "Web", "Reddit"],
    )
    generate_pie_charts(
        os.path.join(images, "reddit_proportion_articles.png"),
        [55, 88, 252],
        ["Reddit", "TikTok", "Web"],
    )
    generate_pie_charts(
        os.path.join(images, "reddit_proportion_threads.png"),
        [1519, 88, 252],
        ["Reddit", "TikTok", "Web"],
    )
    generate_ngrams(threaded_db, tables, source)


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
