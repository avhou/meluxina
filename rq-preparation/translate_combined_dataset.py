import sys
import os.path
import wget
import fasttext
import sqlite3
import re

def detect_language(text, model):
    prediction = model.predict(text, k=1)  # k=1 means top-1 prediction
    lang_code = prediction[0][0].replace('__label__', '')  # Extract language code
    confidence = prediction[1][0]  # Confidence score
    return lang_code, confidence

def translate_combined_datasets(dataset: str):
    if not os.path.exists("lid.176.bin"):
        print("downloading fasttext model")
        wget.download("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", "lid.176.bin")

    model = fasttext.load_model("lid.176.bin")
    with sqlite3.connect(dataset) as conn:
        for r in conn.execute(f"select url, text from articles where detected_language is null or detected_language = '' order by url asc").fetchall():
            url = r[0]
            text = r[1]
            text = re.sub(r'\s+', ' ', text)
            try:
                lang_code, confidence = detect_language(text, model)
                print(f"detected language for {url} is {lang_code} with confidence {confidence}")
                conn.execute(f"update articles set detected_language = ? where url = ?", (lang_code, url))
                conn.commit()
            except:
                print(f"could not detect language for url {url}")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : translate_combined_dataset.py <combined-db.sqlite>")
    translate_combined_datasets(sys.argv[1])
