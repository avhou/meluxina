import argparse
import sqlite3
from llama_index.core.node_parser import SentenceSplitter
from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from models import SentimentScore, SentimentScores


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
    args = parser.parse_args()

    if args.generate_flair:
        print(f"generating sentiment scores using flair for {args.input_db}")
        generate_flair(args.input_db, args.chunk_size, args.chunk_overlap)
    if args.generate_vader:
        print(f"generating sentiment scores using vader for {args.input_db}")
        generate_vader(args.input_db, args.chunk_size, args.chunk_overlap)

    print("done")
