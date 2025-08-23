import re
import heapq
from collections import Counter
import math

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def chunk_text(text, chunk_size=200):
    tokens = preprocess(text)
    return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

def get_tf(tokens):
    tf = Counter(tokens)
    for i in tf:
        tf[i] = tf[i] / len(tokens)
    return tf

def get_idf(documents):
    idf = {}
    num_documents = len(documents)
    all_tokens = set(token for doc in documents for token in doc)
    for token in all_tokens:
        num_documents_with_token = sum(1 for doc in documents if token in doc)
        idf[token] = math.log(num_documents / (1 + num_documents_with_token))
    return idf

def get_tfidf_vector(tokens, idf):
    tfidf_vector = {}
    tf = get_tf(tokens)
    for token in tokens:
        tfidf_vector[token] = tf.get(token, 0) * idf.get(token, 0)
    return tfidf_vector

def get_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def get_top_k_rules(doc_text, rule_texts, rule_filenames, k=5):
    """
    Retrieves the top-k most relevant rule chunks for a given document.

    Args:
        doc_text: The text of the user's document.
        rule_texts: A list of strings, where each string is the content of a rule file.
        rule_filenames: A list of filenames corresponding to the rule_texts.
        k: The number of top rules to retrieve.

    Returns:
        A list of the top-k most relevant rule texts.
    """
    if not rule_texts:
        return []

    all_chunks = []
    for i, rule_text in enumerate(rule_texts):
        # Use larger chunks for rule texts to reduce number of chunks processed
        chunks = chunk_text(rule_text, chunk_size=400)
        for chunk in chunks:
            all_chunks.append((chunk, rule_filenames[i]))

    processed_chunks = [preprocess(chunk[0]) for chunk in all_chunks]
    processed_doc = preprocess(doc_text)

    idf = get_idf(processed_chunks + [processed_doc])
    doc_vec = get_tfidf_vector(processed_doc, idf)

    # Compute similarities and select top-k using a heap for efficiency
    similarity_with_index = []
    for i, chunk_tokens in enumerate(processed_chunks):
        chunk_vec = get_tfidf_vector(chunk_tokens, idf)
        sim = get_cosine_similarity(doc_vec, chunk_vec)
        similarity_with_index.append((sim, i))

    top = heapq.nlargest(k, similarity_with_index, key=lambda x: x[0])
    # Stabilize ordering among ties for consistency
    top.sort(key=lambda x: (-x[0], x[1]))
    top_indices = [idx for _, idx in top]
    top_k_rules = [all_chunks[i][0] for i in top_indices]
    top_k_filenames = [all_chunks[i][1] for i in top_indices]

    print(f"Top {k} relevant rule chunks identified from: {top_k_filenames}")

    return top_k_rules
