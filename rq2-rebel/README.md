# todo

- combined_dataset.sqlite bevat de 3 sources (web, tiktok, reddit), maar zonder de reddit threads in de `articles` table
- combined_dataset_triples_250.sqlite bevat de gegenereerde triples per split van 250 woorden van de combined_dataset.sqlite database

- combined_dataset_reddit.sqlite bevat enkel de reddit source, in thread vorm, in de `articles_reddit` table.
- combined_dataset_reddit_triples_250.sqlite bevat de gegenereerde triples per split van 250 woorden van de combined_dataset_reddit.sqlite database

we willen komen tot een dataset met de 3 sources :

- zonder reddit threads (OK == combined_dataset.sqlite)
- met reddit threads

dit doen we door de 2 andere sources (web, tiktok) te combineren met de reddit threads.

idem voor de triples, maar de verwerking is anders

- zonder reddit threads (OK == combined_dataset_triples_250.sqlite)
- met reddit threads nog aan te maken, maar opgelet uit de combined_dataset_triples_250.sqlite enkel die URLs copieren waar geen reddit in zit, omdat die er te veel in zitten

aanmaken van de datasets in duckdb

```duckdb
attach '/home/alexander/ou/IM9506-AF/database/combined_dataset.sqlite' as non_threaded_data (TYPE sqlite);
attach '/home/alexander/ou/IM9506-AF/database/combined_dataset_reddit.sqlite' as reddit_threads (TYPE sqlite);
attach '/home/alexander/ou/IM9506-AF/database/combined_threaded_dataset.sqlite' as out (TYPE sqlite);
create table out.articles as select * from non_threaded_data.articles where source in ('web', 'tiktok');
insert into out.articles select * from reddit_threads.articles_reddit;
```

aanmaken van de triples db

```duckdb
attach '/home/alexander/ou/IM9506-AF/database/combined_dataset_triples_250.sqlite' as non_threaded_data (TYPE sqlite);
attach '/home/alexander/ou/IM9506-AF/database/combined_dataset_reddit_triples_250.sqlite' as reddit_threads (TYPE sqlite);
attach '/home/alexander/ou/IM9506-AF/database/combined_threaded_dataset_triples_250.sqlite' as out (TYPE sqlite);
create table out.chunked_articles as select * from non_threaded_data.chunked_articles where url not like '%reddit%';
insert into out.chunked_articles select * from reddit_threads.chunked_articles;
```
