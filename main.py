from citations_study.cluster_citations import run_citation_clustering


if __name__ == "__main__":
    corpus_path = "C:/Users/mathew.barber/Documents/openiti/2023-1-8"
    meta_csv = "C:/Users/mathew.barber/Documents/openiti/OpenITI_metadata_2023-1-8.csv"
    run_citation_clustering(corpus_path, meta_csv)