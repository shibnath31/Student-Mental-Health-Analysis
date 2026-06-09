# Student-Mental-Health-Analysis

ABSTRACT:-Student mental health has emerged as a serious global concern, largely driven by increasing academic pressure, competitive learning environments, and personal challenges faced by young adults. Early identification of mental health risks is important for preventing long-term psychological issues, yet traditional detection methods can be slow, subjective, or inaccessible. This study presents a data-driven approach to predict the likelihood of depression among students using machine learning techniques. A comprehensive dataset containing demographic, academic, lifestyle, and psychological indicators—such as gender, age, academic pressure, study satisfaction, sleep duration, financial stress, suicidal thoughts, and family history of mental illness—was analyzed to uncover patterns associated with depression.

Data preprocessing involved removing irrelevant variables, handling class imbalance, converting subjective responses into standardized formats, and transforming sleep duration into a numerical scale. Exploratory data analysis revealed strong associations between depression and factors such as high academic pressure, low study satisfaction, reduced sleep duration, and financial instability. Among all predictive attributes, suicidal thoughts and family history of mental illness were the most influential indicators of depression.

A CatBoost machine learning classifier was selected for its superior handling of categorical variables and its efficiency without extensive preprocessing. The model demonstrated strong predictive performance, achieving an ROC-AUC of approximately 0.92, along with high accuracy and F1-score across stratified k-fold evaluations. The results show that machine learning can identify high-risk students with significant reliability, thus presenting an opportunity for educational institutions to take proactive mental-health initiatives.

Overall, the study highlights the potential of artificial intelligence in detecting depression risk early, promoting timely interventions, and supporting student well-being. Future research can integrate real-time behavioral data and multi-institution datasets to improve generalizability and support the development of mental-health monitoring systems.

Keywords:- Student mental health, Depression prediction, Machine learning, CatBoost classifier, Academic pressure, Financial stress, Study satisfaction, Sleep duration, Suicidal thoughts, Data-driven analysis, Mental health monitoring, Early intervention, Predictive modeling, Educational well-being.

# Step-by-Step Explanation


# Step 1:Import Libraries

We import the required Python libraries such as pandas, numpy, seaborn, matplotlib, sklearn, and catboost.
These help in data loading, visualization, preprocessing, and model training.

# Step 2: Load Dataset

The CSV file of student mental health is loaded using:

df = pd.read_csv("studentdepression.csv")


This dataset contains details about students like academic pressure, study satisfaction, sleep duration, etc.

# Step 3: Check Data Quality

We check:
✔ Missing values
✔ Data types
✔ Basic statistics
This helps us understand whether the data is clean or needs corrections.

# Step 4: Remove Irrelevant Columns

Some columns do not help in depression prediction (example: ID, Profession).
So they are removed to avoid noise in the model.

# Step 5: Data Preprocessing

This includes:

Converting sleep duration text (e.g., "5–6 hours") into a number

Replacing missing values

Encoding categorical features (like Gender, City, etc.)

After preprocessing, the dataset becomes ready for modeling.

# Step 6: Visualization and EDA

Graphs such as bar plots, count plots, and histograms help identify patterns.
For example:

High academic pressure → higher risk of depression

Short sleep duration → higher risk of depression

This step provides insights into the data.

# Step 7: Split Train and Test Data

The dataset is divided into:

80% training data

20% testing data

This helps test the model’s performance on unseen data.

# Step 8: Train CatBoost Model

We use CatBoostClassifier because it handles categorical data automatically.
The model is trained to predict the value of Depression based on other features.

# Step 9: Make Predictions

The model predicts the depression risk for testing data and compares it with the actual result.

# Step 10: Evaluate Model

We calculate:

Accuracy

F1 score

ROC-AUC score

Confusion matrix

These metrics show how well the model can identify students with depression.

# Step 11: Feature Importance

CatBoost shows which features influence prediction the most.
Top indicators include:

Suicidal thoughts

Academic pressure

Financial stress

Sleep duration

Study satisfaction

Step 12: Final Results
<img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/57bdbe5f-a71e-48fc-a471-9d7582b5fedc" />



# The model achieved:

ROC-AUC ≈ 0.92

Accuracy ≈ 85%
It shows that machine learning can effectively detect depression risk among students.
<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/721645ff-8bf0-45d2-8192-d36b4885933d" />


🌟 Final Outcome

This project proves that:
 Mental health prediction using ML is possible
At-risk students can be identified early
Colleges and universities can use such models for early interventions

# Author 
Shibnath Rana
ML Intern|Student of MCA 
Devloped as part of Machine Learning 
