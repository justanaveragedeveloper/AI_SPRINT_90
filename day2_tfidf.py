import numpy as np
import json

# Load job descriptions from JSON
def load_data(file_path='data.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['job_descriptions']

# Create vocabulary from all documents
def create_vocabulary(documents):
    vocab = set()
    for doc in documents:
        words = doc.lower().split()
        vocab.update(words)
    return sorted(list(vocab))

# Compute Term Frequency (TF)
def compute_tf(document, vocabulary):
    words = document.lower().split()
    tf_vector = []
    for word in vocabulary:
        tf = words.count(word) / len(words) if len(words) > 0 else 0
        tf_vector.append(tf)
    return np.array(tf_vector)

# Compute Inverse Document Frequency (IDF) - PROFESSIONAL STANDARD
def compute_idf(documents, vocabulary):
    num_docs = len(documents)
    idf_vector = []
    
    for word in vocabulary:
        # Count how many documents contain this word
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        
        # PROFESSIONAL FORMULA: log((1+n) / (1+df)) + 1
        # This ensures NO negative values and never zeroes out
        idf = np.log((1 + num_docs) / (1 + doc_count)) + 1
        idf_vector.append(idf)
    
    return np.array(idf_vector)

# Compute TF-IDF matrix
def compute_tfidf_matrix(documents, vocabulary):
    tf_matrix = np.array([compute_tf(doc, vocabulary) for doc in documents])
    idf_vector = compute_idf(documents, vocabulary)
    
    # TF-IDF = TF * IDF (broadcasting)
    tfidf_matrix = tf_matrix * idf_vector
    return tfidf_matrix, idf_vector

# Cosine Similarity from Day 1
def cosine_similarity(vector1, vector2):
    """
    Measures the angle between two vectors.
    Returns a score between 0 (different) and 1 (identical direction).
    """
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Prevent division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

# SEARCH ENGINE FUNCTION - The core of talent-iq
def search(query, vocabulary, idf_vector, tfidf_matrix, job_descriptions):
    """
    Search for the most relevant job description based on user query.
    
    Args:
        query: String - user's search query (e.g., "I want a React job")
        vocabulary: List - all unique words from job descriptions
        idf_vector: Array - IDF scores for each word
        tfidf_matrix: 2D Array - TF-IDF vectors for all jobs
        job_descriptions: List - original job texts
    
    Returns:
        Tuple: (best_match_index, similarity_score, best_match_text, all_similarities)
    """
    
    # Step 1: Convert query to TF vector
    query_tf = compute_tf(query, vocabulary)
    
    # Step 2: Apply IDF to get query TF-IDF vector
    query_tfidf = query_tf * idf_vector
    
    # Step 3: Calculate cosine similarity with every job
    similarities = []
    for i, job_vector in enumerate(tfidf_matrix):
        sim = cosine_similarity(query_tfidf, job_vector)
        similarities.append(sim)
    
    # Step 4: Find the best match
    similarities = np.array(similarities)
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    
    return best_match_index, best_score, job_descriptions[best_match_index], similarities

# Display all matches ranked
def display_all_matches(similarities, job_descriptions, query):
    print(f"\n   📊 ALL MATCHES RANKED:")
    # Create list of (index, score) pairs and sort by score descending
    ranked = sorted([(i, similarities[i]) for i in range(len(similarities))], 
                    key=lambda x: x[1], reverse=True)
    
    for rank, (idx, score) in enumerate(ranked, 1):
        if score > 0:
            match_indicator = "🎯" if rank == 1 else "  "
            print(f"   {match_indicator} #{rank}: Score {score:.3f} - {job_descriptions[idx][:50]}...")

# Main execution with interactive loop
def main():
    # Load data
    jobs = load_data()
    
    print("="*70)
    print("TALENT-IQ SEARCH ENGINE (TF-IDF + Cosine Similarity)")
    print("="*70)
    
    print("\n📁 Loaded Job Descriptions:")
    for i, job in enumerate(jobs):
        print(f"   Job {i+1}: {job}")
    
    # Create vocabulary and compute TF-IDF
    vocabulary = create_vocabulary(jobs)
    tfidf_matrix, idf_vector = compute_tfidf_matrix(jobs, vocabulary)
    
    print(f"\n📚 Vocabulary Size: {len(vocabulary)} words")
    
    # Show IDF scores for important words
    print("\n📊 Word Importance Scores (Higher = More Valuable):")
    important_words = ['react', 'python', 'nodejs', 'mongodb', 'ai', 'frontend', 'backend']
    for word in important_words:
        if word in vocabulary:
            idx = vocabulary.index(word)
            print(f"   '{word}': {idf_vector[idx]:.3f}")
    
    print("\n" + "="*70)
    print("🔍 INTERACTIVE JOB SEARCH ENGINE")
    print("="*70)
    print("Commands:")
    print("   - Type your search query (e.g., 'Python developer')")
    print("   - Type 'quit' or 'exit' to stop")
    print("   - Type 'all' to see all jobs")
    print("="*70)
    
    # INTERACTIVE LOOP
    while True:
        print("\n" + "-"*70)
        query = input("🔎 Search for jobs: ").strip()
        
        # Exit conditions
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using Talent-IQ Search Engine. Goodbye!")
            break
        
        # Show all jobs
        if query.lower() == 'all':
            print("\n📋 All available jobs:")
            for i, job in enumerate(jobs, 1):
                print(f"   {i}. {job}")
            continue
        
        # Empty query
        if not query:
            print("   ⚠️ Please enter a search query or type 'quit' to exit")
            continue
        
        # Perform search
        best_idx, score, best_job, all_similarities = search(
            query, vocabulary, idf_vector, tfidf_matrix, jobs
        )
        
        # Display results
        print(f"\n   🏆 TOP MATCH:")
        print(f"   Job {best_idx+1}: {best_job}")
        print(f"   📊 Confidence Score: {score:.3f} ({score*100:.1f}%)")
        
        # Provide recommendation strength
        if score > 0.3:
            print(f"   ✅ STRONG MATCH - Highly recommend this candidate")
        elif score > 0.1:
            print(f"   ⚠️ MODERATE MATCH - Consider reviewing other options")
        elif score > 0:
            print(f"   ❌ WEAK MATCH - Job may not be relevant")
        else:
            print(f"   ❌ NO MATCH FOUND - Try different keywords")
        
        # Show all matches if there are multiple good ones
        if len(jobs) > 1:
            display_all_matches(all_similarities, jobs, query)
        
        # Smart suggestions based on query
        print(f"\n   💡 TIP:", end=" ")
        if score < 0.1 and query:
            print(f"Try using more specific keywords like 'React', 'Python', or 'NodeJS'")
        elif score > 0.3:
            print(f"This is a strong match! Contact this candidate")
        else:
            print(f"Try rephrasing your search or use technical keywords")

# Run the interactive search engine
if __name__ == "__main__":
    main()