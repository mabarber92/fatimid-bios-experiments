"""Search functions for finding citations that contain Sira"""

from utilities.markdown_files import markdownCorpus
import pandas as pd

CITATION_REGEX = r"(\W\w++){0,5}\W*([و]قال|ذكر|يقول)(\W\w++){0,14}\W+[ال]?ل?سير[هة]\W*(\W\w++){5}"
CIT_PHRASE = r"[و]?(?:قال|ذكر|يقول|[يت]ذكر)"
BIO_PHRASE = r"[ال]?ل?سير[هة]"
WORD_PATTERN = r"(?:\W+\w+)"
START_DATE = 0
END_DATE = 1500

def build_citation_regex(cit_phrase, bio_phrase, pre_capture_len=0, mid_capture_len=14, post_capture_len=5):
    """Build a regex to capture possible references to biographies, capturing a set number of 
    tokens before (pre_capture_len), between the citation (cit_phrase) and the biographical marker (bio_phrase)
    (the mid_capture_len), and after the main phrase (post_capture_len). 
    Greedier captures with more context are more likely to capture greater variety of references to biographies,
    but they might be noisier for certain forms of clustering and alignment
    For regex to work - most clean out any markdown (e.g. ~~ and #)"""
    if pre_capture_len == 0:
        full_regex =  rf"\W+{cit_phrase}{WORD_PATTERN}{{0,{mid_capture_len}}}\W+{bio_phrase}{WORD_PATTERN}{{{post_capture_len}}}"
    else:
        full_regex = rf"{WORD_PATTERN}{{0,{pre_capture_len}}}\W+{cit_phrase}{WORD_PATTERN}{{0,{mid_capture_len}}}\W+{bio_phrase}{WORD_PATTERN}{{{post_capture_len}}}"

    return full_regex
    

def search_citations(corpus_path, meta_csv, regex, return_csv=None):
    """Search the corpus for citations matching the regex"""
    corpus = markdownCorpus(corpus_path, meta_csv, start_date=START_DATE, end_date=END_DATE,
                            book_uri_list = None, primary_only=True, multi_process=True)
    results = corpus.search_corpus(regex)
    df = pd.DataFrame(results, columns=["search_result"])
    df = df.drop_duplicates().reset_index(drop=True)
    if return_csv:        
        df.to_csv(return_csv, index=False, encoding="utf-8-sig")
    return df["search_result"].tolist()