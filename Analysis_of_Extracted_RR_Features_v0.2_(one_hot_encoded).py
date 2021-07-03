#!/usr/bin/env python
# coding: utf-8

# In[180]:


import os

import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, GlobalAveragePooling1D
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow_addons as tfa

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, classification_report, make_scorer, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from keras.utils import np_utils

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier


# In[2]:


'''
Read in data and conduct initial pre-processing -> NULL removal strategy, imputation and distribution viewing
'''


# In[3]:


# Read features from csv into df
def read_csv(file_name, file_dir):
    file_path =  file_dir + file_name
    df_features = pd.read_csv(file_path)
    df_features = df_features.iloc[1:] # Shift second row to first row
    df_features.iloc[:,0] = df_features.iloc[:,0].astype('int64') # Change first col (patient id) to be an int
    return df_features


# In[4]:


file_name = 'Results.csv'
file_dir = r'c:/Users/ozzya/PycharmProjects/Enigma/Circadia RR - Features Extraction/'

df_features = read_csv(file_name, file_dir)


# In[5]:


df_features.shape


# In[6]:


# remove any rows that have null outcome
def remove_outcomes_null(df):
    df = df[df['Outcome'].notna()]
    return df


# In[7]:


df_features = remove_outcomes_null(df_features)


# In[8]:


df_features.shape


# In[9]:


# Impute null values in cols
def impute_null (df):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    idf = pd.DataFrame(imp.fit_transform(df)) # impute df
    idf.columns = df.columns
    idf.index = df.index
    idf.isnull().sum()
    return idf


# In[10]:


# Keep columns where majority of values are not null
def remove_majority_null(df, NULL_PORTION):
    # only keep columns with majority non-null values
    df_nonfeat = df.select_dtypes(exclude=['float64']) # non-feature cols i.e patient id and outcome -> not to be masked
    df_feat = df.select_dtypes(include=['float64']) # all feature cols
    mask = df_feat.columns[df_feat.isnull().mean() < NULL_PORTION]
    df_feat = df_feat[mask]
    idf_feat = impute_null(df_feat) # Call impute function
    df = pd.concat([df_nonfeat, idf_feat], axis=1)
    return df


# In[11]:


df_features = remove_majority_null(df_features, NULL_PORTION=0.5)


# In[12]:


df_features.shape


# In[13]:


# Explore for distribution for given col feature
def plot_hist (df, col_name):
    plt.hist(df[col_name])
    plt.title('Distribution of ' + col_name)
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.show()

# # Find list of columns
# for col in df_features.columns:
#     plot_hist(df_features, col)


# In[14]:


'''
Feature selection -> use different methods to determine which features should be selected -> pearson correlation, chi-sqaured, RFE, RF Tree-based
'''


# In[15]:


# Seperate features from labels
def get_features_and_labels (df, label_col):
    X = df.select_dtypes(include=['float64'])
    y = df[label_col]
    return X, y


# In[16]:


X, y = get_features_and_labels (df_features, 'Outcome')


# In[17]:


# Observe distribution of outcome labels across patients
def plot_labels (y):
    plt.bar(y.value_counts().index, y.value_counts().values)
    plt.show()


# In[18]:


plot_labels(y) #observed data looks imbalanced


# In[19]:


# Use chi to calculate which features are highly dependent on response i.e. outcome label
def chi_features (X, y, k):
    chi_selector = SelectKBest(chi2, k=k)
    X_norm = MinMaxScaler().fit_transform(X)
    X_kbest_features_chi = chi_selector.fit_transform(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_features = X.loc[:,chi_support].columns.tolist()
    print(str(len(chi_features)), 'selected features: ', chi_features)
    return chi_support, chi_features


# In[20]:


k = 25 # number of features to select
chi_support, chi_features = chi_features (X, y, k)


# In[21]:


# method -> ranking features by importance, discarding the least important features, and re-fitting the model
def recursive_feature_elimination_features (X, y, k):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=k, step=100, verbose=1)
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_features = X.loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_features)), 'selected features: ', rfe_features)
    return rfe_support, rfe_features


# In[22]:


rfe_support, rfe_features = recursive_feature_elimination_features (X, y, k)


# In[23]:


# select features based on coefficient weights (shrunk feature weights to be removed)
def lasso_logisitc_features(X, y, k):
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=k)
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_features = X.loc[:,embeded_lr_support].columns.tolist()
    print(str(len(embeded_lr_features)), 'selected features', embeded_lr_features)
    return embeded_lr_support, embeded_lr_features


# In[24]:


embeded_lr_support, embeded_lr_features = lasso_logisitc_features(X, y, k)


# In[25]:


# Drive importance of each feature -> derived from how “pure” each of the buckets is.
def random_forrest_tree_features(X, y, k):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=k)
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_rf_selector.fit(X_norm, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_features = X.loc[:,embeded_rf_support].columns.tolist()
    print(str(len(embeded_rf_features)), 'selected features', embeded_rf_features)
    return embeded_rf_support, embeded_rf_features


# In[26]:


embeded_rf_support, embeded_rf_features = random_forrest_tree_features(X, y, k)


# In[27]:


def lightgbm_features(X, y, k):
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
    reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embeded_lgb_selector = SelectFromModel(lgbc, max_features=k)
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lgb_selector.fit(X_norm, y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_features = X.loc[:,embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_features)), 'selected features:  ', embeded_lgb_features)
    return embeded_lgb_support, embeded_lgb_features


# In[28]:


embeded_lgb_support, embeded_lgb_features = lightgbm_features(X, y, k)


# In[29]:


# Display results of selected features across all methods in dataframe
def feature_selection_comparison(X, chi_support, rfe_support, embeded_lr_support, embeded_rf_support, embeded_lgb_support):
    feature_name = X.columns.tolist()
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support, 'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    return feature_selection_df


# In[30]:


feature_selection_comparison = feature_selection_comparison(X, chi_support, rfe_support, embeded_lr_support, embeded_rf_support, embeded_lgb_support)


# In[31]:


feature_selection_comparison.head(25)


# In[32]:


# Based of feature selection method set columns in dataframe
def select_column_features(df, method_type_features):
    df_nonfeat = df.select_dtypes(exclude=['float64']) # non-feature cols i.e patient id and outcome -> not to be masked
    df = df[method_type_features]
    df = pd.concat([df_nonfeat, df], axis=1)
    return df


# In[33]:


df_features.shape


# In[34]:


# Select columns from variables
df_features = select_column_features(df_features, chi_features)


# In[35]:


X, y = get_features_and_labels (df_features, 'Outcome')


# In[36]:


print(y)


# In[37]:


# Convert string labels to numerical
def encode_labels(y):
    global le
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y


# In[223]:


print(y, len(y))


# In[38]:


y = encode_labels(y)
dummy_y = np_utils.to_categorical(y).astype(int) # categories are ordinal and have no natural rank ordering so should be one-hot encoded


# In[39]:


# Split data into test and train and inizialize kf objects for cv
kf = KFold(shuffle=True, random_state=72018, n_splits=3) # Initialize kf object to use for training later


# In[40]:


# Wrap cohen_kapper_score to use for GridSearch scoring mechanism
kappa_scorer = make_scorer(cohen_kappa_score)


# In[41]:


# confusion matrix to show predicted vs actual
def confusion_matrix(y_test, y_pred):
    confusion_matrix = pd.crosstab(y_pred.argmax(axis=1), y_test.argmax(axis=1), rownames=['Predicted'], colnames=['Actual']) 
    return confusion_matrix
#     return sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = 'Reds')
    
# Display classification report for precision, recall and f1 score for all classes
def display_classification_report(y_test, y_pred):
    try:
        return classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=['0: Alert', '1: Death', '2: Home', '3: Hospitalisation', '4: No Outcome'])
    except:
        return classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=['1: Death', '2: Home', '3: Hospitalisation', '4: No Outcome'])


# In[42]:


# Create estimator and pipeline parameter to pass through GridSearchCV
def find_best_model(X, y, model, kf, scorer):
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72018)
    estimator = Pipeline([("scaler", StandardScaler()),
            ("polynomial_features", PolynomialFeatures()),
            ("model", model)
                         ])
    params = {
        'polynomial_features__degree': [1, 2, 3]
    }
    
    grid = GridSearchCV(estimator, params, cv=kf, verbose=0, scoring='accuracy') # can't use scorer as cohen's kappa can only take single input labels
    grid.fit(X_train, y_train) # Fit train data
    if model == MultiOutputClassifier(LogisticRegression()):
        coeffs = grid.best_estimator_.named_steps['model'].coef_
    else:
        coeffs = 'NA'
    y_predict = grid.predict(X_test) # Check results on test set
    model_test_score = cohen_kappa_score(y_predict.argmax(axis=1), y_test.argmax(axis=1))
    model_name = str(grid.estimator[2])
    best_cv_score = grid.best_score_
    best_params = grid.best_params_
    
    return y_predict, y_test, model_test_score, model_name, best_cv_score, best_params


# In[43]:


# Create list of multi-class models to iterate through
models = [MultiOutputClassifier(LogisticRegression()), DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5), DecisionTreeClassifier(criterion = "entropy",
            random_state = 100,max_depth=3, min_samples_leaf=5)]
print(models)


# In[44]:


cm = [] # List to store confusion matrix to later plot for each model
cr = [] # list to store classification report for each model
# Iterate through each model to find best model parameters
for model in models:
    y_predict, y_test, model_test_score, model_name, best_cv_score, best_params = find_best_model(X, dummy_y, model, kf, kappa_scorer)
    print(model_test_score, model_name, best_cv_score, best_params)
    cm.append(confusion_matrix(y_test, y_predict))
    cr.append(display_classification_report(y_test, y_predict))


# In[45]:


# Logistic Regression confusion matrix
print(cr[0])
sns.heatmap(cm[0], annot=True, fmt = 'g', cmap = 'Reds')


# In[46]:


# Descion tree confusion matrix 'gini'
print(cr[1])
sns.heatmap(cm[1], annot=True, fmt = 'g', cmap = 'Reds')


# In[47]:


# Descion tree confusion matrix 'entropy'
print(cr[2])
sns.heatmap(cm[2], annot=True, fmt = 'g', cmap = 'Reds')


# In[ ]:


'''
Build TF DNN model with scikitlearn wrapper
'''


# In[213]:


# define dnn for multi-class classification
def dnn_multiclass_classifier (X, y):
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.CohenKappa(num_classes=5), tfa.metrics.F1Score(num_classes=5, average='macro'), tf.keras.metrics.AUC(multi_label=False)])
    return model


# In[214]:


X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=72018)


# In[215]:


model = dnn_multiclass_classifier (X, dummy_y)


# In[216]:


model.summary()


# In[224]:


history = model.fit(x=X_train, y=y_train, validation_data = (X_test, y_test), epochs=500, batch_size=5, verbose=0)


# In[225]:


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# In[227]:


# Visualize results of model fit - all graphical results can be found on TB
plot_graphs(history, 'f1_score')
plot_graphs(history, 'loss')


# In[ ]:




