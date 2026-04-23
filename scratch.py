from rank_bm25 import BM25Okapi, BM25Plus
corpus = [
    "hello world",
    "hello there",
    "hello everyone",
    "hello friend",
    "world of warcraft"
]
tokenized_corpus = [doc.split() for doc in corpus]
query = "hello world".split()

bm25okapi = BM25Okapi(tokenized_corpus)
print("Okapi:", bm25okapi.get_scores(query))

bm25plus = BM25Plus(tokenized_corpus)
print("Plus:", bm25plus.get_scores(query))
