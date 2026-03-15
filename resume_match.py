from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume = "Python machine learning data analysis NLP"
job_description = "Looking for machine learning engineer with python and NLP"

documents = [resume, job_description]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])

print("Resume Match Score:", similarity[0][0])