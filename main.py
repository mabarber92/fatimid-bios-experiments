from citations_study.cluster_citations import run_citation_clustering


if __name__ == "__main__":
    corpus_path = "E:/OpenITI Corpus/corpus_2023_1_8"
    meta_csv = "E:/Corpus Stats/2023/OpenITI_metadata_2023-1-8.csv"
    run_citation_clustering(corpus_path, meta_csv, existing_results='data/sira_citations.csv')