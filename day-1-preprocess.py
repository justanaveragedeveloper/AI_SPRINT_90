import numpy as np

# 1. Your raw data (Imagine this came from your MongoDB/talent-iq)
sentences = [
    "AI is the future of engineering",           # Job Description 1
    "Engineering is about solving problems",     # Job Description 2
    "I love AI and engineering",                 # User Profile
    "AI AI AI is the future"                     # Spammy/Repetitive resume
]

# 2. Create unique vocabulary
def create_vocabulary(sentences):
    vocab = set()
    for sentence in sentences:
        words = sentence.lower().split()
        vocab.update(words)
    return sorted(list(vocab))  # noqa: C414

# Bag of Words with frequency counting
def sentence_to_frequency(sentence, vocabulary):
    words = sentence.lower().split()
    frequency_vector = [words.count(word) for word in vocabulary]
    return np.array(frequency_vector)

# COSINE SIMILARITY - The professional fix for magnitude bias
def cosine_similarity(vector1, vector2):
    """
    Measures the angle between two vectors, not just their length.
    Returns a score between 0 (completely different) and 1 (identical direction).
    """
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Prevent division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

# Create vocabulary and frequency matrix
vocabulary = create_vocabulary(sentences)
frequency_matrix = []
for sentence in sentences:
    frequency_matrix.append(sentence_to_frequency(sentence, vocabulary))

matrix = np.array(frequency_matrix)

print("="*70)
print("TALENT-IQ JOB MATCHING ENGINE")
print("="*70)
print(f"\nVocabulary ({len(vocabulary)} words):")
print(vocabulary)
print("\nFrequency Matrix (Bag of Words):")
print(matrix)
print("\nSentence Index:")
for i, sentence in enumerate(sentences):
    print(f"  S{i+1}: {sentence}")

print("\n" + "="*70)
print("COMPARISON: Raw Dot Product vs Cosine Similarity")
print("="*70)

# Compare Job Description 1 with everything
job_desc_1 = matrix[0]  # "AI is the future of engineering"
user_profile = matrix[2]  # "I love AI and engineering"
spammy_resume = matrix[3]  # "AI AI AI is the future"

print("\n🎯 JOB MATCHING: 'AI Engineer Position'")
print(f"   Job Description: '{sentences[0]}'")
print(f"   Candidate A: '{sentences[2]}' (Balanced skills)")
print(f"   Candidate B: '{sentences[3]}' (Repetitive/spammy)")
print("\n📊 Raw Dot Product (FLAWED):")
print(f"   Candidate A → Dot: {np.dot(job_desc_1, user_profile)}")
print(f"   Candidate B → Dot: {np.dot(job_desc_1, spammy_resume)}")
print("   ❌ B looks BETTER just because 'AI' is repeated!")

print("\n✅ Cosine Similarity (FIXED):")
cos_a = cosine_similarity(job_desc_1, user_profile)
cos_b = cosine_similarity(job_desc_1, spammy_resume)
print(f"   Candidate A → Cosine: {cos_a:.3f}")
print(f"   Candidate B → Cosine: {cos_b:.3f}")
print("   ✅ Now A and B are properly normalized by length!")

print("\n" + "="*70)
print("FULL SIMILARITY MATRIX (Cosine Similarity)")
print("="*70)

# Calculate cosine similarity between all pairs
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        cos_sim = cosine_similarity(matrix[i], matrix[j])
        dot_sim = np.dot(matrix[i], matrix[j])
        
        print(f"\n📝 S{i+1} ↔ S{j+1}:")
        print(f"   S{i+1}: {sentences[i][:40]}...")
        print(f"   S{j+1}: {sentences[j][:40]}...")
        print(f"   Raw Dot: {dot_sim:3d}  |  Cosine: {cos_sim:.3f}")
        
        # Interpretation for job matching
        if cos_sim > 0.7:
            print(f"   🔥 {int(cos_sim*100)}% Match → EXCELLENT candidate!")
        elif cos_sim > 0.4:
            print(f"   👍 {int(cos_sim*100)}% Match → Good fit")
        elif cos_sim > 0.1:
            print(f"   ⚠️ {int(cos_sim*100)}% Match → Weak match")
        else:
            print(f"   ❌ {int(cos_sim*100)}% Match → Not relevant")
            