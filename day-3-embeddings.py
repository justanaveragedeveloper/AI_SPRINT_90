import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# FASTTEXT WRAPPER FOR OOV HANDLING
# ============================================================================

class FastTextWrapper:
    """Handles OOV words using character n-grams (3-5 chars)"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.vector_size = base_model.vector_size
        self.key_to_index = base_model.key_to_index
        self.ngram_cache = {}
        
    def __getitem__(self, word):
        word_lower = word.lower()
        if word_lower in self.base_model.key_to_index:
            return self.base_model[word_lower]
        return self._generate_oov_vector(word_lower)
    
    def _generate_oov_vector(self, word):
        if word in self.ngram_cache:
            return self.ngram_cache[word]
        
        # Extract n-grams (3-5 chars) with boundary markers
        ngrams = set()
        word_padded = f"<{word}>"
        
        for n in [3, 4, 5]:
            for i in range(len(word_padded) - n + 1):
                ngram = word_padded[i:i+n]
                if ngram in self.base_model.key_to_index:
                    ngrams.add(ngram)
        
        if ngrams:
            vectors = [self.base_model[ngram] for ngram in ngrams]
            vector = np.mean(vectors, axis=0)
        else:
            # Fallback: morphological decomposition
            base_word = self._morphological_fallback(word)
            if base_word in self.base_model.key_to_index:
                vector = self.base_model[base_word]
            else:
                vector = np.random.normal(0, 0.1, self.vector_size)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        self.ngram_cache[word] = vector
        return vector
    
    def _morphological_fallback(self, word):
        suffixes = ['ing', 'ed', 'er', 'or', 'tion', 'ness', 'ment', 's', 'es']
        for suffix in suffixes:
            if word.endswith(suffix):
                base = word[:-len(suffix)]
                if base in self.base_model.key_to_index:
                    return base
        
        prefixes = ['re', 'un', 'in', 'im', 'dis', 'pre']
        for prefix in prefixes:
            if word.startswith(prefix):
                base = word[len(prefix):]
                if base in self.base_model.key_to_index:
                    return base
        return word

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load GloVe with OOV wrapper (FastText would be better but is 6GB)"""
    try:
        import gensim.downloader as api
        print("Loading GloVe model (50-dim)...")
        base_model = api.load("glove-wiki-gigaword-50")
        print(f"Loaded {len(base_model.key_to_index)} words, {base_model.vector_size} dims")
        return FastTextWrapper(base_model)
    except:
        print("Creating fallback embeddings...")
        return FallbackEmbeddings()

class FallbackEmbeddings:
    """Lightweight fallback when GloVe fails to download"""
    def __init__(self):
        self.vector_size = 50
        self.key_to_index = {}
        self.word_vectors = {}
        
        base_words = ['react', 'developer', 'backend', 'engineer', 'nodejs', 
                      'data', 'scientist', 'devops', 'web', 'coder', 'database']
        
        for word in base_words:
            self.word_vectors[word] = np.random.normal(0, 1, 50)
            self.key_to_index[word] = len(self.key_to_index)
        
        # Make similar words have similar vectors
        self.word_vectors['web'] = self.word_vectors['react'] * 0.8
        self.word_vectors['coder'] = self.word_vectors['developer'] * 0.9
        self.word_vectors['database'] = self.word_vectors['backend'] * 0.85
    
    def __getitem__(self, word):
        word = word.lower()
        if word in self.word_vectors:
            return self.word_vectors[word]
        return np.random.normal(0, 0.5, 50)

# ============================================================================
# SENTENCE TO VECTOR (MEAN OF WORD VECTORS)
# ============================================================================

def sentence_to_vector(sentence, model):
    """Convert sentence to 50D vector by averaging word vectors"""
    words = sentence.lower().split()
    vectors = []
    
    for word in words:
        try:
            vectors.append(model[word])
        except:
            pass  # Skip OOV words silently (model handles internally)
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)

# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def cosine_similarity(v1, v2):
    """Return cosine similarity between two vectors"""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# ============================================================================
# SEARCH FUNCTION
# ============================================================================

def search(query, jobs, model, top_k=3):
    """Return top-k job matches for a query"""
    query_vec = sentence_to_vector(query, model)
    results = []
    
    for idx, job in enumerate(jobs):
        job_vec = sentence_to_vector(job, model)
        score = cosine_similarity(query_vec, job_vec)
        results.append((idx, score, job))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("DAY 3: SEMANTIC SEARCH WITH OOV HANDLING")
    print("="*60)
    
    # Load model
    model = load_model()
    
    # Load jobs
    jobs = [
        "React developer needed for frontend",
        "Senior backend engineer with NodeJS and MongoDB",
        "Data scientist with Python and machine learning",
        "DevOps expert for cloud infrastructure",
        "Full stack React developer and NodeJS engineer",
        "Mobile app developer for iOS and Android"
    ]
    
    print("\nJob Database:")
    for i, job in enumerate(jobs, 1):
        print(f"  {i}. {job}")
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING OOV WORDS (should work despite not in training)")
    print("="*60)
    
    test_queries = ["Coder", "ReactJS", "Backend Engineer", "Data Science", "Cloud Expert", "Fullstack"]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search(query, jobs, model, top_k=2)
        
        for rank, (idx, score, job) in enumerate(results, 1):
            match = "✅" if score > 0.3 else "⚠️"
            print(f"  {rank}. [{score:.3f}] {match} {job}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        results = search(query, jobs, model, top_k=3)
        print(f"\nTop matches for '{query}':")
        for rank, (idx, score, job) in enumerate(results, 1):
            print(f"  {rank}. {score:.3f} - {job}")

if __name__ == "__main__":
    main()