# Bank-Management-Sys
A predictive analysis project using a Decision Tree Classifier to forecast term deposit subscriptions. Built with Python, pandas, seaborn, and scikit-learn.
➢ Importing all the necessary libraries:
1.	NUMPY
2.	PANDAS
3.	MATPLOTLIB
4.	SEABORN
5.	SKLEARN
➢ Data Analysis: Data analysis is a process of inspecting, cleansing, transforming, and modelling data with the goal of discovering useful information, informing conclusions, and supporting decision-making. It is the first step to do the project because data exploration is very important to know the inputs.
1.	Graphs: • Box plot is a method for graphically depicting groups of numerical data through their quartiles. • Hist plot is used to represent data provided in a form of square groups. • A violin plot is a method of plotting numeric data, it is similar to a box plot. • Correlation is used to find how one variable is correlated with other. • Pair plot is way to visualize relationship between each variables.it produces a matrix of relationships between variable in your data for an instant examination of our data.
   
➢ Data preprocessing: Data preprocessing is a data mining technique which is used to transform the raw data in a useful and efficient format. Data preprocessing is the process of transforming raw data into an understandable format. The quality of the data should be checked before applying machine learning or data mining algorithms. There are three significant steps in data preprocessing in Machine Learning:
 ✓ Extracting dependent and independent variable.
 ✓ Encoding the categorical data. 
 ✓ Splitting the dataset. 
 ✓ Analyzing several models on the dataset.
 
Extracting dependent and independent variable
-----------------------------------------------
It is important to distinguish the matrix of features independent variables and dependent variables from dataset . In our dataset there are independent variable that are age, education, job and one is the dependent variable Y which have to predict. Encoding the categorical data Machine learning models require all input and output variables to be numeric. This means that if your data contains categorical data, you must encode it to numbers before you can fit and evaluate a model. The two most popular techniques are an Label Encoding and a One-Hot Encoding. Label Encoding refers to converting the labels into a numeric form so as to convert them into the machine-readable form. Machine learning algorithms can then decide in a better way how those labels must be operated. It is an important pre�processing step for the structured dataset in supervised learning. Splitting the dataset The reason is that when the dataset is split into train and test sets, there will not be enough data in the training dataset for the model to learn an effective mapping of inputs to outputs. There will also not be enough data in the test set to effectively evaluate the model performance. The simplest way to split the modelling dataset into training and testing sets is to assign 2/3 data points to the former and the remaining one-third to the latter. Therefore, we train the model using the training set and then apply the model to the test set. In this way, we can evaluate the performance of our model. Separating data into training and testing sets is an important part of evaluating data mining models. Because the data in the testing set already contains known values for the attribute that you want to predict, it is easy to determine whether the model's guesses are correct.
Training Set: A subset of dataset to train the machine learning model, and we already know the output. Test set: A subset of dataset to test the machine learning model, and by using the test set, model predicts the output. Nevertheless, common split percentages include: • Train: 80%, Test: 20% • Train: 70%, Test: 30% • Train: 50%, Test: 50% Analyzing several models on the dataset ✓ Import Decision Tree from sklearn library and train the model and predicted the class labels for Xtest and ytest. ✓ Then from sklearn.metrics we have to import:
CLASSIFICATION REPORT: It gives the precision ,recall, f1 score and support from the ytest and ypred. Precision: It predicts the positive out of all the positive. Recall: It predicts positive out of all the actual positive. F1 score: It represents the model score of precision and recall.
CONFUSION MATRIX: To evaluate the accuracy of classification.
ACCURACY SCORE: Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition: Accuracy = Number of correct predictions Total number of predictions
Result: I have predicted by Decision Tree that 1119 people have subscribed and 7924 people have not subscribed the term deposit.
Performance: The Performance of model is 86%


