#Name - Ayush Dhoundiyal
#Homework 2
# Importing necessary libraries for data manipulation, machine learning, and visualization
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Loading the pre-trained logistic regression model (BEST_LR_MODEL_EVER.pkl)
with open('BEST_LR_MODEL_EVER.pkl', 'rb') as f:
    clf = pickle.load(f) 
    
    # Reading the 'test_bias_new.csv' file
    df = pd.read_csv('test_bias_new.csv')
    
    # Separating features (X) and target (Y)
    X = df.loc[:, df.columns != 'approved']
    Y = df['approved']
    
    # Normalizing feature values column-wise
    norm_X = preprocessing.normalize(X, axis=0)
    
    # Checking the model score on this dataset
    print("New Test Score:", clf.score(norm_X, Y))

# Reading the old test data from 'test_bias.csv'
df_old = pd.read_csv('test_bias.csv')

# Extracting features and target from the old DataFrame
X_old = df_old.loc[:, df.columns != 'approved']
Y_old = df_old['approved']

# Normalizing old test data
norm_X_old = preprocessing.normalize(X_old, axis=0)
print("New Test Score:", clf.score(norm_X_old, Y_old))

# Filtering the DataFrame by sex=0 and checking model performance
df_modified = df[df['sex'] == 0]
df_modified
X_modified = df_modified.loc[:, df_modified.columns != 'approved']
Y_modified = df_modified['approved']
norm_X_modified = preprocessing.normalize(X_modified, axis=0)
print("New Test Score:", clf.score(norm_X_modified, Y_modified))

# Filtering the DataFrame by sex=1 and checking model performance
df_modified = df[df['sex'] == 1]
X_modified = df_modified.loc[:, df_modified.columns != 'approved']
Y_modified = df_modified['approved']
norm_X_modified = preprocessing.normalize(X_modified, axis=0)
print("New Test Score:", clf.score(norm_X_modified, Y_modified))

# Display the first rows of df_old
df_old.head()

# Function to calculate bias by filtering data based on 'sex' column
def calculating_bias(clf, file_to_load, sex_column):
    # Reads a CSV file and filters rows where 'sex' equals 'sex_column'
    df = pd.read_csv(file_to_load)
    df_modified = df[df['sex'] == sex_column]
    
    # Separate features from target
    X_modified = df_modified.loc[:, df_modified.columns != 'approved']
    Y_modified = df_modified['approved']
    
    # Normalize the features
    norm_X_modified = preprocessing.normalize(X_modified, axis=0)
    
    # Predict using the loaded classifier
    Y_pred = clf.predict(norm_X_modified)
    
    # Generate the confusion matrix
    cm = confusion_matrix(Y_modified, Y_pred, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Print test score and confusion matrix results
    print("New Test Score:", clf.score(norm_X_modified, Y_modified))
    print("Confusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Calling 'calculating_bias' on different subsets of data
calculating_bias(clf, 'test_bias_new.csv', 0)
calculating_bias(clf, 'test_bias_new.csv', 1)
calculating_bias(clf, 'test_bias.csv', 1)
calculating_bias(clf, 'test_bias.csv', 0)

# Function to demonstrate bias by filtering data on both sex and approved status
def proving_bias(clf, file_to_load, sex_column, approval):
    # Reads CSV data
    df = pd.read_csv(file_to_load)
    
    # Filters data for specific sex and approval status
    df_modified = df[(df['sex'] == sex_column) & (df['approved'] == approval)]
    
    # Separate features and target
    X_modified = df_modified.loc[:, df_modified.columns != 'approved']
    Y_modified = df_modified['approved']
    
    # Normalize filtered features
    norm_X_modified = preprocessing.normalize(X_modified, axis=0)
    
    # Predict using the loaded model
    Y_pred = clf.predict(norm_X_modified)
    
    # Generate confusion matrix
    cm = confusion_matrix(Y_modified, Y_pred, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Print the score and confusion matrix
    print("New Test Score:", clf.score(norm_X_modified, Y_modified))
    print("Confusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Proving bias by checking combinations of sex and approval in the new test data
proving_bias(clf, 'test_bias_new.csv', 1, 0)
proving_bias(clf, 'test_bias_new.csv', 1, 1)
proving_bias(clf, 'test_bias_new.csv', 0, 1)
proving_bias(clf, 'test_bias_new.csv', 0, 0)

################################################################################
################################################################################
################################################################################
# Now Second Case Study
################################################################################
################################################################################
################################################################################

# Function to load the 'PassPredictionLR.pkl' model and evaluate it on a specified CSV
def read_file_initialy_and_get_result(file_name):
    with open('PassPredictionLR.pkl', 'rb') as f:
        clf = pickle.load(f)
        
        # Reading data from the file
        df = pd.read_csv(file_name)
        
        # Separating features and target
        X = df.loc[:, df.columns != 'will_pass']
        Y = df['will_pass']
        
        # Normalizing features
        norm_X = preprocessing.normalize(X, axis=0)
        
        # Getting predictions and confusion matrix
        Y_pred = clf.predict(norm_X)
        cm = confusion_matrix(Y, Y_pred, labels=[0, 1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Printing the score and confusion matrix
        print("New Test Score:", clf.score(norm_X, Y))
        print("Confusion Matrix:")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Testing the model on the train and test CSVs
read_file_initialy_and_get_result('passPred_train.csv')
read_file_initialy_and_get_result('passPred_test.csv')

# Reading the training CSV for direct inspection
df = pd.read_csv('passPred_train.csv')
df.head()

# Splitting data by class 0 and class 1
df_will_pass_0 = df[df['will_pass'] == 0]
df_will_pass_1 = df[df['will_pass'] == 1]
df_will_pass_0
df_will_pass_1

# Function to load the model, print coefficients, then evaluate
def read_file_initialy_and_get_additioal_result(file_name):
    with open('PassPredictionLR.pkl', 'rb') as f:
        clf = pickle.load(f)
    df = pd.read_csv(file_name)
    X = df.loc[:, df.columns != 'will_pass']
    Y = df['will_pass']
    
    # If the classifier has coefficients, print them out
    if hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]
        feature_names = X.columns.tolist()
        print("\n=== Model Coefficients ===")
        feature_weights = clf.coef_[0]
        print("Feature Weights:")
        for feature, weight in zip(feature_names, feature_weights):
            print(f"{feature}: {weight:.4f}")
        print(f"\nIntercept: {intercept:.4f}")
    else:
        print("This model does not expose 'coef_' or 'intercept_' (not a linear model).")
    
    # Normalize and predict
    norm_X = preprocessing.normalize(X, axis=0)
    Y_pred = clf.predict(norm_X)
    cm = confusion_matrix(Y, Y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Print performance results
    print("New Test Score:", clf.score(norm_X, Y))
    print("Confusion Matrix:")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Checking additional results on the training file
read_file_initialy_and_get_additioal_result('passPred_train.csv')

# Inspecting examples of will_pass == 1 and will_pass == 0
df = pd.read_csv('passPred_train.csv')
df[df['will_pass'] == 1]
df[df['will_pass'] == 0]

# Printing class distribution in the training data
train_df = pd.read_csv('passPred_train.csv')
print("Class Distribution in Training Data:")
print(train_df['will_pass'].value_counts())

# Re-loading the logistic regression model to examine predicted probabilities
train_df = pd.read_csv('passPred_train.csv')
with open('PassPredictionLR.pkl', 'rb') as f:
    clf = pickle.load(f)
X = train_df.drop(columns=['will_pass'])
y = train_df['will_pass']

# Normalizing the training data
X_norm = preprocessing.normalize(X, axis=0)

# Getting predicted probabilities for class 1
probs = clf.predict_proba(X_norm)[:, 1]

# Printing min, max, and mean predicted probabilities
print("Min probability:", probs.min())
print("Max probability:", probs.max())
print("Mean probability:", probs.mean())

# Plotting a histogram of predicted probabilities
plt.hist(probs, bins=20)
plt.xlabel('Predicted Probability for Class 1')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities on Training Data')
plt.savefig('./histogram.png')
plt.show()

# Box plots for each feature by class
for feature in train_df.columns:
    if feature != 'will_pass':
        sns.boxplot(x='will_pass', y=feature, data=train_df)
        plt.title(f'Distribution of {feature} by Class')
        plt.savefig(f'./box_plot{feature}.png')
        plt.show()

# Converting normalized features to a DataFrame for inspection
normalized_df = pd.DataFrame(X_norm, columns=X.columns)
result_df = pd.concat([normalized_df, y], axis=1)
print("Normalized Training Data with Target (first 5 rows):")
print(result_df.head())
print("\nSummary Statistics of Normalized Data:")
print(normalized_df.describe())

# Inspecting rows belonging to class 1 and class 0 in the normalized data
result_df[result_df['will_pass'] == 1].head(15)
result_df[result_df['will_pass'] == 0].head(15)

# Function to re-train the model using StandardScaler to mitigate bias
def retraing_the_model(file_name):
    df = pd.read_csv(file_name)
    X = df.drop(columns=['will_pass'])
    y = df['will_pass']
    
    # Creating a pipeline for scaling + logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42))
    ])
    
    # Training on the entire data
    pipeline.fit(X, y)
    
    # Checking performance on the same data
    y_pred = pipeline.predict(X)
    train_accuracy = accuracy_score(y, y_pred)
    cm_train = confusion_matrix(y, y_pred, labels=[0, 1])
    
    # Printing results
    print("Training Accuracy:", train_accuracy)
    print("Training Confusion Matrix:\n", cm_train)
    tn, fp, fn, tp = cm_train.ravel()
    print("\nDetailed Confusion Matrix Values:")
    print("True Negatives (TN):", tn)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)
    print("True Positives (TP):", tp)
    
    # Returning the trained pipeline for later use
    return pipeline

# Re-training the model on passPred_train.csv with StandardScaler
pipeline = retraing_the_model('passPred_train.csv')

# Function to predict on a test file using the newly trained pipeline
def predicting_true_function(file_name, pipeline):
    test_df = pd.read_csv(file_name)
    X_test = test_df.drop(columns=['will_pass'])
    y_test = test_df['will_pass']
    
    # Predicting using the pipeline
    y_pred = pipeline.predict(X_test)
    
    # Calculating accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Generating a confusion matrix
    cm_test = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Test Accuracy:", test_accuracy)
    print("Test Confusion Matrix:\n", cm_test)

# Evaluating the newly trained model on the test data
predicting_true_function('passPred_test.csv', pipeline)

# Checking scaled data for additional insight
X_scaled = pipeline.named_steps['scaler'].transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df['will_pass'] = y.values
print("Scaled Data with Target:")
print(X_scaled_df.head())

# Inspecting scaled rows belonging to each class
X_scaled_df[X_scaled_df['will_pass'] == 0].head(15)
X_scaled_df[X_scaled_df['will_pass'] == 1].head(15)