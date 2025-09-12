from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

def initialiseEmbedModel(model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix"):
    """Use AutoModel and AutoTokenizer to load a model from the HuggingFace hub"""
    print("loading model...")       
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_model = AutoModel.from_pretrained(model_name,  output_hidden_states=True)

    return tokenizer, transformer_model

def compute_embeddings(sentence, tokenizer, transformer_model):
    """Take a sequence compute and return the offsets and embeddings for that sequence"""
    # Tokenize the input sequence and get the individual input tokens as strings
    tokens = tokenizer([sentence], return_offsets_mapping=True, return_tensors="pt")
    tokens_as_strings = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    offsets = tokens.pop('offset_mapping')  # Remove offset mapping to avoid issues in model input
    
    # Compute the embeddings for the tokens and extract the hidden states from the specified layer
    embeddings = transformer_model(**tokens, output_hidden_states=True)

    return offsets[0].tolist(), tokens_as_strings, embeddings

def select_layer_clean(offsets, tokens_as_strings, embeddings, hidden_layer, tokenizer):
    """Select a hidden layer from embeddings and clean out the special tokens
    return the cleaned embeddings and corresponding offsets as lists"""
    layer_embs = embeddings.hidden_states[hidden_layer][0]

    # Remove special tokens and their embeddings
    cleaned_embeds = []
    cleaned_offsets = []
    special_tokens = tokenizer.all_special_tokens
    for token, offset, embedding in zip(tokens_as_strings, offsets, layer_embs):
        if token in special_tokens:
            continue
        else:
            cleaned_embeds.append(embedding.detach().cpu().numpy())
            cleaned_offsets.append(offset)
    
    return cleaned_offsets, cleaned_embeds

def embed_and_tokenise(sentence, tokenizer, transformer_model, hidden_layer=-1):
    """Embed a sentence using the specified model and return a pair of sub-tokens"""
    
    # Compute the embeddings and offsets for the sequence
    offsets, tokens_as_strings, embeddings = compute_embeddings(sentence, tokenizer, transformer_model)

    cleaned_offsets, cleaned_embeds = select_layer_clean(offsets, tokens_as_strings, embeddings, hidden_layer, tokenizer)
    
    return cleaned_offsets, cleaned_embeds

def compute_cut(offsets, embeds, original_sequence, threshold=0.5, logging=False):
    """Using the embeddings for each token, compute the optimum cut point in the sequence
    This is a placeholder function that will be tuned in tune_metrics.py
    Note for the cases that this is tuned on we know that the sequence starts in the right place, we're looking
    for the end. Moreover, we know that the sequence will at least be of length 2 - so we use the distance between
    the first two sub-tokens as our baseline and measure the drop in similarity from there."""

    # Compute the consine similarity between the first two tokens
    base_sim = cosine_similarity([embeds[0]], [embeds[1]])[0][0]

    cut = False
    token_pos = 1
    while not cut:
        # Check the next token exists
        if token_pos + 1 >= len(embeds):
            cut = True
            cut_pos = len(embeds) # Don't cut the sequence - take original length'
        elif (base_sim - cosine_similarity([embeds[token_pos]], [embeds[token_pos+1]])[0][0]) > threshold:
            cut = True
            if logging:
                print("Cut at token position ", token_pos)
            cut_pos = token_pos + 1 # Cut after this token
        else:
            token_pos += 1
    
    # Now we have the cut position, we use it to find the offset boundary and walk until we hit a space
    cut_offset = offsets[cut_pos][1]  # End of the token at the cut position
    while cut_offset < len(original_sequence) and original_sequence[cut_offset] != " ":
        cut_offset += 1
    
    return original_sequence[:cut_offset]
    

def compute_sequence_cut(sequence, tokenizer, transformer_model, threshold=0.1, hidden_layer=-1):
    """Using sequence embeddings, identify the optimum way to cut a sequence
    This will be tuned in tune_metrics.py to cut at the ends of citations, but could be tuned for other tasks"""

    offsets, embeds = embed_and_tokenise(sequence, tokenizer, transformer_model, hidden_layer)


    # Use function to compute where the cut should be made
    cut_sequence = compute_cut(offsets, embeds, sequence, threshold)

    return cut_sequence


def test_pipeline(sequence):
    
    tokenizer, transformer_model = initialiseEmbedModel()
    cut_sequence = compute_sequence_cut(sequence, tokenizer, transformer_model)
    print(cut_sequence)

if __name__ == "__main__":
    test_sequence = " قال جامع سيرة الوزير اليازوري وقصر النيل بمصر"
    test_pipeline(test_sequence)