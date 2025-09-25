from citations_study.cluster_citations import run_citation_clustering
from utilities.search_funcs import build_citation_regex, search_citations

CIT_PHRASE = r"[و]?(?:قال|ذكر|يقول|[يت]ذكر)"
BIO_PHRASE = r"[ال]?ل?سير[هة]"

if __name__ == "__main__":
    corpus_path = "E:/OpenITI Corpus/corpus_2023_1_8"
    meta_csv = "E:/Corpus Stats/2023/OpenITI_metadata_2023-1-8.csv"
    # run_citation_clustering(corpus_path, meta_csv, existing_results='data/sira_citations.csv')

    regex = build_citation_regex(CIT_PHRASE, BIO_PHRASE, post_capture_len=10)
    found = search_citations(corpus_path, meta_csv, regex, return_csv="data/sira_citations_long.csv")