import argparse
import sqlite3
from llama_index.core.node_parser import SentenceSplitter
from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from itertools import product

from models import SentimentScore, SentimentScores, PromptTemplate, PromptTemplates
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import urllib.request
from collections import defaultdict


def flair_scores(db: str, chunk_size: int, chunk_overlap: int) -> str:
    is_threaded = "threaded" in db
    return (
        f"rq3-flair-scores-threaded-chunk-size-{chunk_size}-overlap-{chunk_overlap}.json"
        if is_threaded
        else f"rq3-flair-scores-chunk-size-{chunk_size}-overlap-{chunk_overlap}.json"
    )


def vader_scores(db: str, chunk_size: int, chunk_overlap: int) -> str:
    is_threaded = "threaded" in db
    return (
        f"rq3-vader-scores-threaded-chunk-size-{chunk_size}-overlap-{chunk_overlap}.json"
        if is_threaded
        else f"rq3-vader-scores-chunk-size-{chunk_size}-overlap-{chunk_overlap}.json"
    )


def roberta_scores(db: str, chunk_size: int, chunk_overlap: int) -> str:
    is_threaded = "threaded" in db
    return (
        f"rq3-roberta-scores-threaded-chunk-size-{chunk_size}-overlap-{chunk_overlap}.json"
        if is_threaded
        else f"rq3-roberta-scores-chunk-size-{chunk_size}-overlap-{chunk_overlap}.json"
    )


def prompt(db: str) -> str:
    is_threaded = "threaded" in db
    return "rq3-prompts-threaded.json" if is_threaded else "rq3-prompts.json"


def generate_vader(db: str, chunk_size: int = 50, chunk_overlap: int = 5):
    print(f"processing db {db}")
    nltk.download("vader_lexicon")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    analyzer = SentimentIntensityAnalyzer()

    with sqlite3.connect(db) as conn:
        scores_by_disinformation = {}
        sentiment_scores = []
        for url, text, disinformation in conn.execute("select url, translated_text, disinformation from articles order by url"):
            disinformation = 0 if disinformation == "n" else 1
            print(f"processing url {url}, disinformation {disinformation}")
            sentences = splitter.split_text(text)

            scores = []
            for sentence in sentences:
                score = analyzer.polarity_scores(sentence)["compound"]
                scores.append(score)

            average_score = np.mean(scores)
            print(f"avg score {average_score:.4f}")
            scores_by_disinformation.setdefault(disinformation, []).append(average_score)
            sentiment_scores.append(SentimentScore(url=url, score=average_score))
        print(f"average score for NO disinformation : {np.mean(scores_by_disinformation[0]):.4f}")
        print(f"average score for disinformation : {np.mean(scores_by_disinformation[1]):.4f}")

        with open(vader_scores(db, chunk_size, chunk_overlap), "w") as f:
            sentiment_scores = SentimentScores(
                scores=sentiment_scores,
                avg_score_disinformation=np.mean(scores_by_disinformation[1]),
                avg_score_no_disinformation=np.mean(scores_by_disinformation[0]),
            )
            f.write(sentiment_scores.model_dump_json())


def generate_flair(db: str, chunk_size: int = 50, chunk_overlap: int = 5):
    print(f"processing db {db}")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    classifier = TextClassifier.load("en-sentiment")

    with sqlite3.connect(db) as conn:
        scores_by_disinformation = {}
        sentiment_scores = []
        for url, text, disinformation in conn.execute("select url, translated_text, disinformation from articles order by url"):
            disinformation = 0 if disinformation == "n" else 1
            print(f"processing url {url}, disinformation {disinformation}")
            sentences = splitter.split_text(text)

            scores = []
            for s in sentences:
                sentence = Sentence(s)
                classifier.predict(sentence)
                label = sentence.labels[0]
                score = label.score if label.value == "POSITIVE" else -label.score
                scores.append(score)

            average_score = np.mean(scores)
            print(f"avg score {average_score:.4f}")
            scores_by_disinformation.setdefault(disinformation, []).append(average_score)
            sentiment_scores.append(SentimentScore(url=url, score=average_score))
        print(f"average score for NO disinformation : {np.mean(scores_by_disinformation[0]):.4f}")
        print(f"average score for disinformation : {np.mean(scores_by_disinformation[1]):.4f}")

        with open(flair_scores(db, chunk_size, chunk_overlap), "w") as f:
            sentiment_scores = SentimentScores(
                scores=sentiment_scores,
                avg_score_disinformation=np.mean(scores_by_disinformation[1]),
                avg_score_no_disinformation=np.mean(scores_by_disinformation[0]),
            )
            f.write(sentiment_scores.model_dump_json())


def generate_roberta(db: str, chunk_size: int = 50, chunk_overlap: int = 5):
    print(f"processing db {db}")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    task = "sentiment"
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.model_max_length = 512
    # download label mapping
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    with sqlite3.connect(db) as conn:
        scores_by_disinformation = {}
        sentiment_scores = []
        for url, text, disinformation in conn.execute("select url, translated_text, disinformation from articles order by url"):
            disinformation = 0 if disinformation == "n" else 1
            print(f"processing url {url}, disinformation {disinformation}")
            sentences = splitter.split_text(text)

            scores = []
            for s in sentences:
                encoded_input = tokenizer(s, return_tensors="pt", truncation=True, max_length=512)
                output = model(**encoded_input)
                model_scores = output[0][0].detach().numpy()
                model_scores = softmax(model_scores)
                ranking = np.argsort(model_scores)
                ranking = ranking[::-1]
                weights = {"negative": -1, "neutral": 0, "positive": 1}
                sentiment_score = sum(model_scores[i] * weights[labels[i]] for i in range(len(labels)))
                scores.append(sentiment_score)

            average_score = np.mean(scores)
            print(f"avg score {average_score:.4f}")
            scores_by_disinformation.setdefault(disinformation, []).append(average_score)
            sentiment_scores.append(SentimentScore(url=url, score=average_score))
        print(f"average score for NO disinformation : {np.mean(scores_by_disinformation[0]):.4f}")
        print(f"average score for disinformation : {np.mean(scores_by_disinformation[1]):.4f}")

        with open(roberta_scores(db, chunk_size, chunk_overlap), "w") as f:
            sentiment_scores = SentimentScores(
                scores=sentiment_scores,
                avg_score_disinformation=np.mean(scores_by_disinformation[1]),
                avg_score_no_disinformation=np.mean(scores_by_disinformation[0]),
            )
            f.write(sentiment_scores.model_dump_json())


def read_scores(threaded: bool, model: str, chunk_size: int, chunk_overlap: int) -> SentimentScores:
    with open(f"rq3-{model}-scores-{'' if not threaded else 'threaded-'}chunk-size-{chunk_size}-overlap-{chunk_overlap}.json") as f:
        return SentimentScores.model_validate_json(f.read())


def generate_table():
    with open("rq3-sentiment-table.md", "w") as f:
        f.write("| model | threaded | chunk_size | chunk_overlap | avg score no disinformation | avg score disinformation | \n")
        f.write("| :---- | :------- | ---------: | ------------: | --------------------------: | -----------------------: | \n")
        for model, threaded, (chunk_size, chunk_overlap) in product(["flair", "vader", "roberta"], [False, True], zip([200, 100, 50], [20, 10, 5])):
            scores = read_scores(threaded, model, chunk_size, chunk_overlap)
            f.write(
                f"| {model} | {'Yes' if threaded else 'No'} | {chunk_size} | {chunk_overlap} | {scores.avg_score_no_disinformation:.4f} | {scores.avg_score_disinformation:.4f} |\n"
            )


def generate_prompts(db: str, chunk_size: int, chunk_overlap: int):
    scores_flair = read_scores("threaded" in db, "flair", chunk_size, chunk_overlap)
    scores_vader = read_scores("threaded" in db, "vader", chunk_size, chunk_overlap)
    scores_roberta = read_scores("threaded" in db, "roberta", chunk_size, chunk_overlap)

    url_to_score = defaultdict(list)
    for score in scores_flair.scores:
        url_to_score[score.url].append(score.score)
    for score in scores_vader.scores:
        url_to_score[score.url].append(score.score)
    for score in scores_roberta.scores:
        url_to_score[score.url].append(score.score)

    templates = []
    with sqlite3.connect(db) as conn:
        for url, text, disinformation in conn.execute("select url, translated_text, disinformation from articles order by url"):
            print(f"processing url {url}, disinformation {disinformation}")
            templates.append(
                PromptTemplate(
                    url=url,
                    article_text=text,
                    ground_truth_disinformation=disinformation,
                    scores=url_to_score[url],
                )
            )
    with open(prompt(db), "w") as f:
        f.write(PromptTemplates(templates=templates).model_dump_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")

    parser.add_argument("--input-db", type=str, required=True, help="Path to the input database file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        required=False,
        help="The chunk size for the sentence splitter",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=5,
        required=False,
        help="The chunk overlap for the sentence splitter",
    )

    parser.add_argument("--generate-flair", action="store_true", help="Flag to generate sentiment scores using flair")
    parser.add_argument("--generate-vader", action="store_true", help="Flag to generate sentiment scores using vader")
    parser.add_argument("--generate-roberta", action="store_true", help="Flag to generate sentiment scores using roberta")
    parser.add_argument("--generate-table", action="store_true", help="Flag to generate score table")
    parser.add_argument("--generate-prompts", action="store_true", help="Flag to generate prompts")
    args = parser.parse_args()

    if args.generate_flair:
        print(f"generating sentiment scores using flair for {args.input_db}")
        generate_flair(args.input_db, args.chunk_size, args.chunk_overlap)
    if args.generate_vader:
        print(f"generating sentiment scores using vader for {args.input_db}")
        generate_vader(args.input_db, args.chunk_size, args.chunk_overlap)
    if args.generate_roberta:
        print(f"generating sentiment scores using vader for {args.input_db}")
        generate_roberta(args.input_db, args.chunk_size, args.chunk_overlap)
    if args.generate_table:
        print("generating table")
        generate_table()
    if args.generate_prompts:
        print(f"generating prompts for {args.input_db}")
        generate_prompts(args.input_db, args.chunk_size, args.chunk_overlap)

    print("done")
