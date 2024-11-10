import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data():
    jobs_data = pd.DataFrame({
        'job_id': range(1, 11),
        'title': ['Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer', 'Marketing Specialist',
                  'Frontend Developer', 'Backend Engineer', 'Data Analyst', 'DevOps Engineer', 'AI Researcher'],
        'description': [
            'Develop software applications using Python and JavaScript',
            'Analyze large datasets and build machine learning models',
            'Manage product lifecycle and work with cross-functional teams',
            'Design user interfaces and improve user experience',
            'Create marketing campaigns and analyze their performance',
            'Build responsive web applications using React and Vue',
            'Develop scalable backend systems using Java and Spring',
            'Perform data analysis and create insightful visualizations',
            'Manage cloud infrastructure and implement CI/CD pipelines',
            'Research and develop cutting-edge AI algorithms'
        ],
        'skills': [
            'Python, JavaScript, Git',
            'Python, R, SQL, Machine Learning',
            'Agile, Jira, Product Strategy',
            'Figma, Sketch, User Research',
            'SEO, Social Media, Analytics',
            'React, Vue, CSS',
            'Java, Spring, Microservices',
            'SQL, Python, Tableau',
            'AWS, Docker, Kubernetes',
            'TensorFlow, PyTorch, NLP'
        ],
        'location': ['New York', 'San Francisco', 'Chicago', 'Seattle', 'Boston',
                     'Austin', 'Los Angeles', 'Denver', 'Atlanta', 'Washington DC'],
        'salary': [100000, 120000, 110000, 95000, 85000, 90000, 105000, 95000, 115000, 130000]
    })
    
    users_data = pd.DataFrame({
        'user_id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'skills': [
            'Python, JavaScript, React',
            'Python, R, Data Analysis',
            'Product Management, Agile, SQL',
            'Java, Spring, Microservices',
            'Machine Learning, TensorFlow, NLP'
        ],
        'preferred_location': ['New York', 'San Francisco', 'Chicago', 'Seattle', 'Boston']
    })
    
    return jobs_data, users_data

def engineer_features(jobs_data, users_data):
    jobs_data['text_features'] = jobs_data['title'] + ' ' + jobs_data['description'] + ' ' + jobs_data['skills']
    users_data['text_features'] = users_data['skills']
    
    tfidf = TfidfVectorizer(stop_words='english')
    job_tfidf = tfidf.fit_transform(jobs_data['text_features'])
    user_tfidf = tfidf.transform(users_data['text_features'])
    
    scaler = MinMaxScaler()
    jobs_data['normalized_salary'] = scaler.fit_transform(jobs_data[['salary']])
    
    all_locations = set(jobs_data['location']) | set(users_data['preferred_location'])
    
    job_locations = pd.get_dummies(jobs_data['location'], prefix='loc')
    for loc in all_locations:
        if f'loc_{loc}' not in job_locations.columns:
            job_locations[f'loc_{loc}'] = 0
    
    user_locations = pd.get_dummies(users_data['preferred_location'], prefix='loc')
    for loc in all_locations:
        if f'loc_{loc}' not in user_locations.columns:
            user_locations[f'loc_{loc}'] = 0
    
    location_columns = sorted(list(all_locations))
    job_locations = job_locations.reindex(columns=[f'loc_{loc}' for loc in location_columns], fill_value=0)
    user_locations = user_locations.reindex(columns=[f'loc_{loc}' for loc in location_columns], fill_value=0)
    
    job_features = np.hstack([job_tfidf.toarray(), job_locations, jobs_data[['normalized_salary']]])
    user_features = np.hstack([user_tfidf.toarray(), user_locations, np.zeros((len(users_data), 1))])  # Add a dummy column for salary
    
    return job_features, user_features, jobs_data, users_data

def knn_recommendation(job_features, user_features, k=5):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(job_features)
    distances, indices = knn.kneighbors(user_features)
    return distances, indices

def content_based_filtering(job_features, user_features):
    similarity_matrix = cosine_similarity(user_features, job_features)
    return similarity_matrix

def collaborative_filtering(jobs_data, users_data):
    interaction_matrix = np.zeros((len(users_data), len(jobs_data)))
    for i, (_, user) in enumerate(users_data.iterrows()):
        for j, (_, job) in enumerate(jobs_data.iterrows()):
            skill_match = len(set(user['skills'].split(', ')).intersection(set(job['skills'].split(', '))))
            location_match = int(user['preferred_location'] == job['location'])
            interaction_matrix[i, j] = skill_match + location_match
    return interaction_matrix

def train_random_forest(job_features, user_features, interactions):
    X = np.concatenate([np.repeat(job_features, len(user_features), axis=0),
                        np.tile(user_features, (len(job_features), 1))], axis=1)
    y = interactions.flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return rf

def get_recommendations(content_similarity, collaborative_similarity, knn_indices, rf_model, job_features, user_features, jobs_data, user_index, top_n=3):
    cb_cf_scores = 0.4 * content_similarity[user_index] + 0.3 * collaborative_similarity[user_index]
    
    knn_scores = np.zeros(len(jobs_data))
    knn_scores[knn_indices[user_index]] = 1 / (np.arange(len(knn_indices[user_index])) + 1)
    
    rf_input = np.concatenate([job_features, np.tile(user_features[user_index], (len(job_features), 1))], axis=1)
    rf_scores = rf_model.predict_proba(rf_input)[:, 1]
    
    combined_scores = 0.4 * cb_cf_scores + 0.3 * knn_scores + 0.3 * rf_scores
    
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    recommendations = jobs_data.iloc[top_indices]
    
    return recommendations

def main():
    jobs_data, users_data = load_data()
    
    job_features, user_features, jobs_data, users_data = engineer_features(jobs_data, users_data)
    
    content_similarity = content_based_filtering(job_features, user_features)
    collaborative_similarity = collaborative_filtering(jobs_data, users_data)
    knn_distances, knn_indices = knn_recommendation(job_features, user_features)
    
    rf_model = train_random_forest(job_features, user_features, collaborative_similarity)
    
    for i, user in users_data.iterrows():
        print(f"\nRecommendations for {user['name']}:")
        recommendations = get_recommendations(content_similarity, collaborative_similarity, knn_indices, rf_model, job_features, user_features, jobs_data, i)
        print(recommendations[['title', 'location', 'salary']])


main()
