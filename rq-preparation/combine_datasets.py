import sys
import duckdb
import os.path


def combine_datasets(news_outlets: str, tiktok: str, reddit: str, output: str):
    print(f"combining news outlets {news_outlets}, tiktok {tiktok}, reddit {reddit}")

    with duckdb.connect() as conn:
        statements = [
            f"""ATTACH '{news_outlets}' as news_outlets (TYPE sqlite);""",
            f"""ATTACH '{tiktok}' as tiktok (TYPE sqlite);""",
            f"""ATTACH '{reddit}' as reddit (TYPE sqlite);""",
            f"""ATTACH '{output}' as output (TYPE sqlite);""",
            f"""drop table if exists output.articles""",
            f"""create table output.articles as 
                select 'web' as source,
                       url,
                       STRPTIME(timestamp, '%Y%m%d%H%M%S') as timestamp, 
                       json_object('host', host, 'political_party', political_party) as metadata,
                       '' as detected_language,
                       content as text, 
                       translated_text, 
                       keywords, 
                       case when relevant = 1 then 'y' else 'n' end as relevant, 
                       case when disinformation = 1 then 'y' else 'n' end as disinformation,
                from news_outlets.outlet_hits 
                where relevant = 1""",
            f"""insert into output.articles
            select 'tiktok' as source,
                   url,
                   STRPTIME(create_time, '%Y-%m-%dT%H:%M:%SZ') as timestamp, 
                   json_object('id', id, 'video_duration', video_duration) as metadata,
                   '' as detected_language,
                   transcription as text, 
                   translated_text, 
                   keywords, 
                   relevant, 
                   case when disinformation = 'y' then 'y' else 'n' end as disinformation,
            from tiktok.video_hits
            where relevant = 'y'""",
            f"""insert into output.articles
            select 'reddit' as source,
                   url,
                   STRPTIME(created, '%Y-%m-%d %H:%M:%S%z') as timestamp, 
                   json_object('id', id, 'subreddit', subreddit) as metadata,
                   '' as detected_language,
                   text, 
                   translated_text, 
                   keywords, 
                   relevant, 
                   case when disinformation = 'y' then 'y' else 'n' end as disinformation,
            from reddit.reddit_hits
            where relevant = 'y'""",

        ]
        for statement in statements:
            print(statement)
            conn.execute(statement)


if __name__ == "__main__":
    if len(sys.argv) <= 4:
        raise RuntimeError("usage : combine_datasets.py <news_outlets-db.sqlite> <tiktok-db.sqlite> <reddit-db.sqlite> <output-db.sqlite>")
    combine_datasets(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
