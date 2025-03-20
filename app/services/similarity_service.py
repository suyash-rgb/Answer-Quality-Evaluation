from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityService:

    def calculate_jaccard_similarity(self, sentence1: str, sentence2: str) -> float:
        set1 = set(sentence1.lower().split())
        set2 = set(sentence2.lower().split())

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0  # Handle empty sets
        return intersection / union

    # def calculate_cosine_similarity_tfidf(self, sentence1: str, sentence2: str) -> float:
    #     vectorizer = TfidfVectorizer()
    #     tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    #     print(tfidf_matrix.toarray()) # Print the TF-IDF vectors
    #     cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    #     return cosine_sim
    
    def calculate_cosine_similarity_tfidf(self, sentence1: str, sentence2: str) -> float:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2)) # Add stop_words='english'
        print("Vectorizer Parameters: ", vectorizer.get_params())
        tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
        print(tfidf_matrix.toarray())  # Print the TF-IDF vectors
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return cosine_sim
