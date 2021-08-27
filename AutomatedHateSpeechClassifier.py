# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 06:34:24 2021

@author: Thean Jun Chao (0127122)
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
from PIL import Image
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
#!pip install tweet-preprocessor
import preprocessor as p
stop_words = stopwords.words('english')
#from sklearn.metrics import roc_curve, roc_auc_score
#Search why use this library instead of the traditional way (faster)
#Use for tokenization, remove stopwords, etc.


def clean_text(text):
    # Remove special characters using the regular expression library
    import re
    # Set up punctuations we want to be replaced (Punctuations, Tags, Symbols)
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(\&)")
    REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
    
    # Send to tweet_processor
    tmp1 = p.clean(text)

    # Remove puctuation & convert all tweets to lower cases
    tmp1 = REPLACE_NO_SPACE.sub("", tmp1.lower())
    tmp1 = REPLACE_WITH_SPACE.sub(" ", tmp1)

    # Tokenization
    tmp2 = nltk.word_tokenize(tmp1)

    # Remove stop words
    tmp2 = [word for word in tmp2 if not word in stop_words]

    #Lemmatization (remove ing, s,etc.)
    clean_text=''
    for word in tmp2:
        clean_text = clean_text + ' ' + str(lemmatizer.lemmatize(word))
    return clean_text

def classifier_evaluation(y_pred, y_test):
    # Confusion Matrix & Classification Report
    # from sklearn.metrics import confusion_matrix
    fig, ax = plt.subplots()
    confusion_matrix = pd.crosstab(y_pred, y_test, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap = 'Blues')
    st.write("Confusion Matrix:")
    st.write(fig)
    
    #matrix = classification_report(y1_pred, y_test)
    #st.write("#### Classification Report :")
    #st.text(matrix)
    st.text('Model Report:\n ' + classification_report(y_pred, y_test))
    
    
df = pd.read_csv('C://Users/Admin/Desktop/GitHub/KDDM-DST-Assignment/labeled_data.csv')

# Data Cleaning
del df['Unnamed: 0']
del df['count']
del df['hate_speech']
del df['offensive_language']
del df['neither']

# Data Cleaning
clean_tweets = []

for index, row in df.iterrows():
    temp_sentence = row['tweet']
    temp_sentence = clean_text(temp_sentence)
    clean_tweets.append(temp_sentence)

df['clean_tweet'] = clean_tweets   

# Add a binary classification column
new_class = []

for index, row in df.iterrows():
    temp_class = row['class']
    if temp_class > 1:
        temp_class = 1
    new_class.append(temp_class)
    
df['binary_class'] = new_class

# Rename the 'class' column
df.rename(columns={'class':'trinary_class'}, inplace = True)

#image = Image.open("C:/Users/Admin/Pictures/hate speech.png")
#st.image(image, width = 750)

st.header('**Hate Speech Classification**')
st.write('---')

menu = st.sidebar.selectbox("Select a Function", ("Profiling Report", "Speech Classification Models"))

if menu == "Profiling Report":
    pr = ProfileReport(df, explorative=True)
    st.header('*Pandas Profiling Report*')
    st_profile_report(pr)
        
if menu == "Speech Classification Models":  
    st.header('*Model Evaluation*')
    # Train & Test Data Split
    x = df['clean_tweet'].astype(str)
    y1 = df['binary_class'].astype(str)
    y2 = df['trinary_class'].astype(str)
    
    x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y1, y2, test_size = 0.2, random_state = 42)
    
    # Vectorize Tweets using TF-IDF
    #from sklearn.feature_extraction.text import CountVectorizer
    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(df['clean_tweet'].astype(str))
    x_train_tfidf = tfidf_vect.transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)
    

    if st.checkbox('Evaluate The Binary Classification Model (Hate, Non-Hate)'):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        svm_binary = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
        svm_binary.fit(x_train_tfidf, y1_train)

        # predict the labels on validation dataset
        y1_pred = svm_binary.predict(x_test_tfidf)
        # Classifier Evaluation
        classifier_evaluation(y1_pred, y1_test)
        
        #User Input
        st.write("""##### Try it out yourself!""")
        binary_text = st.text_area("Classify Using The Binary Model:", "Enter Text")  
        #Clean the Text
        binary_text = clean_text(binary_text)
        
        if st.checkbox('Apply Binary Model'):
            # Preparation for Classifying User Input
            binary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', svm_binary)])
            
            
            # Generate Result
            result = binary_model.predict([binary_text])
            
            if result.astype(int) == 0:
                result_text = "Hate Speech"
            else:
                result_text = "Non-Hate Speech"
                
            st.write(" ##### Result: ", result_text)
           
            # Interpretation of Result
            st.write("""#### Result Interpretation:""")
            binary_model.predict_proba([binary_text])
            binary_explainer = LimeTextExplainer(class_names={"Hate":0, "Non-Hate":1})
            max_features = x_train.str.split().map(lambda x: len(x)).max()
            
            
            random.seed(13)
            idx = random.randint(0, len(x_test))
            
            bin_exp = binary_explainer.explain_instance(
                binary_text, binary_model.predict_proba, num_features=max_features
                )
            
            components.html(bin_exp.as_html(), height=800)
                
    if st.checkbox('Evaluate The Trinary Classification Model (Hate, Offensive, Neither)'):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        svm_trinary = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
        svm_trinary.fit(x_train_tfidf, y2_train)

        # predict the labels on validation dataset
        y2_pred = svm_trinary.predict(x_test_tfidf)
        # Classifier Evaluation
        classifier_evaluation(y2_pred, y2_test)
        
        #User Input
        st.write("""##### Try it out yourself!""")
        #User Input
        trinary_text = st.text_area("Classify Using The Trinary Model:", "Enter Text")  
        #Clean the Text
        trinary_text = clean_text(trinary_text)
        
        if st.checkbox('Apply Trinary Model'):
            # Preparation for Classifying User Input
            trinary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', svm_trinary)])
            
            # Generate Result
            result = trinary_model.predict([trinary_text])
            
            if result.astype(int) == 0:
                result_text = "Hate Speech"
            elif result.astype(int) == 1:
                result_text = "Offensive Language"
            else:
                result_text = "Neither Hate Nor Offensive"
                
            st.write(" ##### Result: ", result_text)
           
            # Interpretation of Result
            st.write("""#### Result Interpretation:""")
            trinary_model.predict_proba([trinary_text])
            trinary_explainer = LimeTextExplainer(class_names={"Hate":0, "Offensive":1, "Neither":2})
            max_features = x_train.str.split().map(lambda x: len(x)).max()
            
            random.seed(13)
            idx = random.randint(0, len(x_test))

            tri_exp = trinary_explainer.explain_instance(
                trinary_text, trinary_model.predict_proba, num_features=max_features, labels=[0, 2]
            )
            
            components.html(tri_exp.as_html(), height=800)