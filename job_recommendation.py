import pandas as pd
import numpy as np
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_kaggle_data():

    path = kagglehub.dataset_download("ravindrasinghrana/job-description-dataset")
    dataset_path = f"{path}/job_descriptions.csv" 
    jobs_data = pd.read_csv(dataset_path)
    

    jobs_data = jobs_data.fillna("Unknown")
    

    irrelevant_columns = [
        "Job Id", "Contact Person", "Contact", "Job Portal", "Company Profile",
        "Job Posting Date", "Preference", "latitude", "longitude"
    ]
    jobs_data = jobs_data.drop(columns=irrelevant_columns)
    

    jobs_data = jobs_data.rename(columns={"Job Title": "title", "Job Description": "description", "Skills": "skills",
                                          "Location": "location", "Salary Range": "salary"})
    
    users_data = pd.DataFrame({
        'user_id': range(1, 11),
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 
                'Frank', 'Grace', 'Hannah', 'Ian', 'Judy'],
        'skills': [
            'Python, JavaScript, React',           
            'Python, R, Data Analysis',         
            'Product Management, Agile, SQL',      
            'Java, Spring, Microservices',           
            'Machine Learning, TensorFlow, NLP', 
            'JavaScript, Node.js, MongoDB',         
            'Digital Marketing, SEO, Analytics',     
            'Graphic Design, Photoshop, Figma',      
            'AWS, Docker, Kubernetes',               
            'C++, Robotics, Embedded Systems'        
        ],
        'preferred_location': [
            'New York', 'San Francisco', 'Chicago', 
            'Seattle', 'Boston', 'Austin', 'Los Angeles', 
            'Denver', 'Atlanta', 'Washington DC'
        ],
        'experience': [
            2, 3, 5, 4, 6, 1, 3, 4, 5, 2
        ],
        'qualifications': [
            'Bachelor\'s in Computer Science',       
            'Master\'s in Data Science',         
            'MBA in Product Management',             
            'Bachelor\'s in Software Engineering',   
            'Ph.D. in AI',                           
            'Associate\'s in Web Development',    
            'Bachelor\'s in Marketing',            
            'Bachelor\'s in Graphic Design',        
            'Bachelor\'s in Cloud Computing',        
            'Master\'s in Robotics',          
        ],
        'salary_expectation': [
            95000, 100000, 110000, 105000, 130000, 
            80000, 85000, 90000, 120000, 115000
        ],
        'availability': [
            'Full-time', 'Part-time', 'Full-time', 'Contract', 
            'Full-time', 'Part-time', 'Contract', 'Full-time', 
            'Full-time', 'Contract'
        ]
    })
    
    return jobs_data, users_data

def preprocess_salary_column(data):
    def extract_salary(salary_str):
        salary_str = salary_str.replace('$', '').replace('K', '000')
        if '-' in salary_str:
            low, high = salary_str.split('-')
            return (float(low) + float(high)) / 2  
        else:
            return float(salary_str)  
    
    data['salary'] = data['salary'].apply(extract_salary)
    return data

def engineer_features(jobs_data, users_data):

    jobs_data['text_features'] = jobs_data['title'] + ' ' + jobs_data['description'] + ' ' + jobs_data['skills']
    users_data['text_features'] = users_data['skills']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    job_tfidf = tfidf.fit_transform(jobs_data['text_features'])
    user_tfidf = tfidf.transform(users_data['text_features'])
    

    scaler = MinMaxScaler()
    jobs_data['normalized_salary'] = scaler.fit_transform(jobs_data[['salary']].fillna(0))
    
    all_locations = set(jobs_data['location']) | set(users_data['preferred_location'])
    job_locations = pd.get_dummies(jobs_data['location'], prefix='loc')
    user_locations = pd.get_dummies(users_data['preferred_location'], prefix='loc')
    
    location_columns = sorted(list(all_locations))
    job_locations = job_locations.reindex(columns=[f'loc_{loc}' for loc in location_columns], fill_value=0)
    user_locations = user_locations.reindex(columns=[f'loc_{loc}' for loc in location_columns], fill_value=0)
    
    job_features = np.hstack([job_tfidf.toarray(), job_locations, jobs_data[['normalized_salary']]])
    user_features = np.hstack([user_tfidf.toarray(), user_locations, np.zeros((len(users_data), 1))]) 
    
    return job_features, user_features, jobs_data, users_data


def knn_recommendation(job_features, user_features, k=5):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(job_features)
    distances, indices = knn.kneighbors(user_features)
    return distances, indices


def content_based_filtering(job_features, user_features):
    return cosine_similarity(user_features, job_features)


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
    return jobs_data.iloc[top_indices]


def evaluate_approaches(jobs_data, users_data, content_similarity, collaborative_similarity, knn_distances, knn_indices, rf_model, job_features, user_features):
    print("\n--- Evaluation Results ---")
    
    print("\nContent-Based Filtering Recommendations:")
    for i, user in users_data.iterrows():
        top_indices = content_similarity[i].argsort()[-3:][::-1]
        recommendations = jobs_data.iloc[top_indices]
        print(f"\nRecommendations for {user['name']}:")
        print(recommendations[['title', 'location', 'salary', 'description']])
    
    # print("\nCollaborative Filtering Recommendations:")
    # for i, user in users_data.iterrows():
    #     top_indices = collaborative_similarity[i].argsort()[-3:][::-1]
    #     recommendations = jobs_data.iloc[top_indices]
    #     print(f"\nRecommendations for {user['name']}:")
    #     print(recommendations[['title', 'location', 'salary']])
    
    # print("\nKNN-Based Recommendations:")
    # for i, user in users_data.iterrows():
    #     top_indices = knn_indices[i]
    #     recommendations = jobs_data.iloc[top_indices]
    #     print(f"\nRecommendations for {user['name']}:")
    #     print(recommendations[['title', 'location', 'salary']])

    # print("\nHybrid Recommendations (Random Forest + Combined Scores):")
    # for i, user in users_data.iterrows():
    #     recommendations = get_recommendations(content_similarity, collaborative_similarity, knn_indices, rf_model, job_features, user_features, jobs_data, i)
    #     print(f"\nRecommendations for {user['name']}:")
    #     print(recommendations[['title', 'location', 'salary']])

def get_recommendations_for_user(user_input):
    jobs_data, users_data = load_kaggle_data()
    jobs_data = preprocess_salary_column(jobs_data)
    
    new_user = pd.DataFrame({
        'user_id': [len(users_data) + 1],
        'name': [user_input['name']],
        'skills': [user_input['skills']],
        'preferred_location': [user_input['preferred_location']],
        'experience': [user_input['experience']],
        'qualifications': [user_input['qualifications']],
        'salary_expectation': [user_input['salary_expectation']],
        'availability': [user_input['availability']]
    })
    users_data = pd.concat([users_data, new_user], ignore_index=True)
    
    job_features, user_features, jobs_data, users_data = engineer_features(jobs_data, users_data)
    content_similarity = content_based_filtering(job_features, user_features)
    collaborative_similarity = collaborative_filtering(jobs_data, users_data)
    knn_distances, knn_indices = knn_recommendation(job_features, user_features)
    rf_model = train_random_forest(job_features, user_features, collaborative_similarity)
    
    recommendations = get_recommendations(
        content_similarity, 
        collaborative_similarity, 
        knn_indices, 
        rf_model, 
        job_features, 
        user_features, 
        jobs_data, 
        len(users_data) - 1
    )
    
    # Return additional fields
    return recommendations[['title', 'location', 'salary', 'experience', 'qualifications', 'availability']]



def main():
    jobs_data, users_data = load_kaggle_data()
    jobs_data = jobs_data.sample(n=5000, random_state=42) 
    jobs_data = preprocess_salary_column(jobs_data)
    job_features, user_features, jobs_data, users_data = engineer_features(jobs_data, users_data)
    content_similarity = content_based_filtering(job_features, user_features)
    collaborative_similarity = collaborative_filtering(jobs_data, users_data)
    knn_distances, knn_indices = knn_recommendation(job_features, user_features)
    rf_model = train_random_forest(job_features, user_features, collaborative_similarity)
    
    evaluate_approaches(jobs_data, users_data, content_similarity, collaborative_similarity, knn_distances, knn_indices, rf_model, job_features, user_features)

if __name__ == "__main__":
    main()
