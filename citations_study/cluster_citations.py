"""Code to extract citations of texts containing 'Sira' and cluster them based on different forms of 
similarity"""
from utilities.search_funcs import build_citation_regex, search_citations
import pandas as pd
import os
import json

CITATION_REGEX = r"(\W\w++){0,5}\W*([و]قال|ذكر|يقول)(\W\w++){0,14}\W+[ال]?ل?سير[هة]\W*(\W\w++){5}"
CIT_PHRASE = r"[و]?(?:قال|ذكر|يقول|[يت]ذكر)"
BIO_PHRASE = r"[ال]?ل?سير[هة]"
WORD_PATTERN = r"(?:\W+\w+)"
START_DATE = 0
END_DATE = 1500

def initialiseEmbedModel(model_name, seqLength=512):
    from sentence_transformers import SentenceTransformer, models
    from torch import nn
    from torch import cuda
    print("loading model...")       
    word_embedding_model = models.Transformer(model_name, seqLength)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                            out_features=seqLength, 
                            activation_function=nn.Tanh())
    # It seems we may not need the device set up
    device = "cuda:0" if cuda.is_available() else "cpu"
    print(device)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)
    print("model loaded")
    return model    


def embed_and_cluster(found_spans, model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix", json_out=None):
    """Embed the spans with sequence embeddings and cluster them using BERTopic (so we have posssible labels for
    clusters of citations). Output a json of cluster labels and aligned sequences"""

    from bertopic import BERTopic


    model = initialiseEmbedModel(model_name, seqLength=25)
    embeddings = model.encode(found_spans, show_progress_bar=True)

    topic_model = BERTopic(language="arabic", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(found_spans, embeddings)

    clustered = topic_model.get_topic_info()
    topic_dict = {}
    for topic_num in clustered["Topic"]:
        topic_dict[topic_num] = {}
        topic_dict[topic_num]["label"] = str(clustered[clustered["Topic"]==topic_num]["Name"].values[0])
        topic_dict[topic_num]["members"] = [found_spans[i] for i, t in enumerate(topics) if t == topic_num]
        topic_dict[topic_num]["size"] = int(clustered[clustered["Topic"]==topic_num]["Count"].values[0])

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(topic_dict, f, ensure_ascii=False, indent=4)
    
    topic_count = len(topic_dict)
    print(f"Identified {topic_count} clusters of citations")
    outlier_count = len(topic_dict[-1]["members"])
    print(f"Of which {outlier_count} were outliers")

    return topic_dict

def run_citation_clustering(corpus_path, meta_csv, existing_results=None, existing_topics=None):
    """Run the citation clustering analysis"""
    # Check for a data directory for storing outputs and intermediaries
    if not os.path.exists("data"):
        os.mkdir("data")
    regex = build_citation_regex(CIT_PHRASE, BIO_PHRASE)
    if not existing_results:
        search_results = search_citations(corpus_path, meta_csv, regex, return_csv="data/sira_citations.csv")
        print(len(search_results))
    else:
        search_results = pd.read_csv(existing_results)["search_result"].tolist()
    
    if not existing_topics:
        topics_dict = embed_and_cluster(search_results, json_out="data/sira_citation_clusters.json")
    else:
        with open(existing_topics, "r", encoding="utf-8") as f:
            topics_dict = json.load(f)   
