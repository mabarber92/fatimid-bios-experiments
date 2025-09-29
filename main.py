# from citations_study.cluster_citations import run_citation_clustering
# from utilities.search_funcs import build_citation_regex, search_citations
from measuring_biographies.count_bio_dates import fetch_process_date_data, analyse_by_label
import pandas as pd
from utilities.markdown_files import mARkdownFile

CIT_PHRASE = r"[و]?(?:قال|ذكر|يقول|[يت]ذكر)"
BIO_PHRASE = r"[ال]?ل?سير[هة]"

if __name__ == "__main__":
    # corpus_path = "E:/OpenITI Corpus/corpus_2023_1_8"
    # meta_csv = "E:/Corpus Stats/2023/OpenITI_metadata_2023-1-8.csv"
    # # run_citation_clustering(corpus_path, meta_csv, existing_results='data/sira_citations.csv')

    # regex = build_citation_regex(CIT_PHRASE, BIO_PHRASE, post_capture_len=10)
    # found = search_citations(corpus_path, meta_csv, regex, return_csv="data/sira_citations_long_files.csv", return_book=True)

    book_file = "data/texts/0845Maqrizi.Muqaffa.Sham19Y0145334-ara1.completed"
    # date_boundaries = [{"label": "pre-Fatimid", "date_range": [1,357]},
    #                    {"label": "Fatimid", "date_range": [358,567]},
    #                    {"label": "Ayyubid", "date_range": [568,648]},
    #                    {"label": "Bahri Mamluk", "date_range": [649,783]},
    #                    {"label": "Circassian Mamluk", "date_range": [784,900]}]
    
    # fetch_process_date_data(book_file, date_boundaries, "data/muqaffa_bio_dates/")
    
    # csv_file = "data/muqaffa_bio_dates/heading_lengths_by_label.csv"
    # df = pd.read_csv(csv_file)
    # analyse_by_label(df, "data/muqaffa_bio_dates", write_summary=False, lengths_above=500)

    markdown_file = mARkdownFile(book_file)
    test_regex = r"الوزيري"
    matching_splits = markdown_file.search_for_header(test_regex, level=1, get_length=True, identify_ms=True, bio_header=True)
    print(matching_splits)