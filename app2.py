import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('best_model.pkl')

# Define the prediction function
def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Define the Streamlit app
def main():
    st.title('Thyroid Outlier Classification')
    
    st.sidebar.header('User Input Features')
    
    # Define the input fields in the sidebar
    Age = st.sidebar.slider('Age', 0.0, 1.0, 0.01)
    Sex = st.sidebar.radio('Sex', [0, 1], index=0, format_func=lambda x: 'Male' if x == 0 else 'Female')
    on_thyroxine = st.sidebar.selectbox('On Thyroxine', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    query_on_thyroxine = st.sidebar.selectbox('Query on Thyroxine', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    on_antithyroid_medication = st.sidebar.selectbox('On Antithyroid Medication', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    sick = st.sidebar.selectbox('Sick', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    pregnant = st.sidebar.selectbox('Pregnant', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    thyroid_surgery = st.sidebar.selectbox('Thyroid Surgery', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    I131_treatment = st.sidebar.selectbox('I131 Treatment', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    query_hypothyroid = st.sidebar.selectbox('Query Hypothyroid', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    query_hyperthyroid = st.sidebar.selectbox('Query Hyperthyroid', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    lithium = st.sidebar.selectbox('Lithium', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    goitre = st.sidebar.selectbox('Goitre', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    tumor = st.sidebar.selectbox('Tumor', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    hypopituitary = st.sidebar.selectbox('Hypopituitary', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    psych = st.sidebar.selectbox('Psych', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    TSH = st.sidebar.slider('TSH', 0.0, 494.0, 1.0)
    T3_measured = st.sidebar.slider('T3 Measured', 0.0, 95.0, 1.0)
    TT4_measured = st.sidebar.slider('TT4 Measured', 0.0, 395.0, 1.0)
    T4U_measured = st.sidebar.slider('T4U Measured', 0.0, 233.0, 1.0)
    FTI_measured = st.sidebar.slider('FTI Measured', 0.0, 642.0, 1.0)
    
    features = [
        Age, Sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, 
        sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid,
        query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, 
        TSH, T3_measured, TT4_measured, T4U_measured, FTI_measured
    ]
    
    if st.sidebar.button('Predict'):
            result = predict(features)
            if result == 0:
                st.sidebar.markdown("<p>Outlier label: <span style='color: green;'>Normal</span></p>", unsafe_allow_html=True)
            else:
                st.sidebar.markdown("<p>Outlier label: <span style='color: red;'>Outlier</span></p>", unsafe_allow_html=True)
    
    # Additional charts
    st.header("Additional Charts")
    
    # Load the data for charts
    data = pd.read_excel('annthyroid_.xlsx')  # Update with your data file path
    data.drop(5659, inplace=True)
    
    # 1. Histogram for Age Ranges
    st.subheader("Age Range Distribution")
    bins = np.linspace(0, 1, 11)  # 10 bins between 0 and 1
    labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    data['Age Range'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    age_hist = data['Age Range'].value_counts().sort_index()
    st.bar_chart(age_hist)
    
    # 2. Pie Chart for Sex
    st.subheader("Sex Distribution")
    sex_counts = data['Sex'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sex_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)
    
    # 3. Count of Pregnant Women
    st.subheader("Count of Pregnant Women")
    pregnant_counts = data['pregnant'].value_counts()
    st.bar_chart(pregnant_counts)
    
    # 4. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
    
    # 5. Box Plot for TSH Levels by Sex
    st.subheader("TSH Levels by Sex")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Sex', y='TSH', data=data, ax=ax3)
    ax3.set_xticklabels(['Male', 'Female'])
    st.pyplot(fig3)

if __name__ == '__main__':
    main()
