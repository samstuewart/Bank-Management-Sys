#Bank-Management-Sys
A predictive analysis project utilizing a Decision Tree Classifier to forecast customer subscription behavior for term deposits. The project demonstrates end-to-end data processing, visualization, and model evaluation using Python and libraries like pandas, seaborn, and scikit-learn.

Key Features
---------------
Data Analysis:
Performed exploratory data analysis (EDA) with visualizations like box plots, histograms, violin plots, correlation matrices, and pair plots to understand relationships between variables.

Data Preprocessing:
Transformed raw data by encoding categorical variables using Label Encoding.
Split the dataset into training and testing subsets for better model evaluation.

Model Training and Evaluation:
Trained a Decision Tree Classifier on customer data to predict term deposit subscriptions.
Evaluated the model using metrics like classification reports, confusion matrices, and accuracy scores.

Steps in the Project
-------------------------------
Data Exploration:
Identified patterns and trends using visualizations.
Examined unique values and handled categorical variables.

Preprocessing:
Extracted independent (age, education, job, etc.) and dependent (y) variables.
Encoded categorical data into numeric form.
Split data into training (80%) and testing (20%) sets.

Model Evaluation:
Classification Report: Assessed precision, recall, F1-score, and support.
Confusion Matrix: Analyzed model predictions.
Accuracy Score: Measured overall model performance.

Results
---------------------------
Predictions:
1119 customers subscribed, while 7924 customers did not subscribe to term deposits.
Performance: Achieved an accuracy of 86% on the dataset.

Technologies and Tools
----------------------------------
Python
Libraries: pandas, numpy, matplotlib, seaborn, sklearn
