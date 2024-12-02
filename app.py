import streamlit as st
from job_recommendation import get_recommendations_for_user

def main():
    st.title("Job Recommendation App")
    st.sidebar.header("User Input")
    
    # User Inputs
    user_input = {
        'name': st.sidebar.text_input("Name", "Enter your name"),
        'skills': st.sidebar.text_area("Skills", "e.g., Python, Machine Learning"),
        'preferred_location': st.sidebar.text_input("Preferred Location", "e.g., New York"),
        'experience': st.sidebar.slider("Years of Experience", 0, 20, 2),
        'qualifications': st.sidebar.text_input("Qualifications", "e.g., Bachelor's in Computer Science"),
        'salary_expectation': st.sidebar.number_input("Salary Expectation", value=90000),
        'availability': st.sidebar.selectbox("Availability", ["Full-time", "Part-time", "Contract"]),
    }
    
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            recommendations = get_recommendations_for_user(user_input)
        
        st.subheader(f"Job Recommendations for {user_input['name']}")
        for _, row in recommendations.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(f"**Location:** {row['location']}")
            st.write(f"**Skills Required:** {row['skills']}")
            st.write(f"**Salary:** ${row['salary']}")
            st.write("---")

if __name__ == "__main__":
    main()
