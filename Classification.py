from copyreg import pickle
from dataclasses import dataclass
from turtle import tiltangle
from typing import Container
import streamlit as st 
import pandas as pd
import pickle 
from tensorflow import keras
import nltk
import json
from nltk.corpus import stopwords
import re 
import numpy as np 
import tensorflow as tf
from matplotlib import pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



#constant 

st.set_page_config(page_title='Classification',layout='wide')

st.title("Welcome to The App")

column1,column2 = st.columns([0.4,0.6])
with column1:
    title = st.text_area('Enter Your Title',placeholder='Computer vision in work....')
    keyword = st.text_area('Enter Your Keywords',placeholder='computer,deeplearning')
    abstract = st.text_area('Enter Your Abstract', placeholder='The aim of this paper is...')




def cleantext(sentence):
    lem = nltk.wordnet.WordNetLemmatizer()
    stopword = stopwords.words('english')
    sno = nltk.stem.SnowballStemmer('english')
    words = [word for word in sentence.split() if word not in stopword]
    sentence = ' '.join(words) #this needs a space to keep the words intact
    sentence = ''.join(sno.stem(sentence))
    sentence = ''.join(lem.lemmatize(sentence)) 
    return re.sub('[^A-Za-z]+', ' ',sentence)


def predictor(models:tf.keras.Model,text:list,n_classes:int,tag_type,data_type:str)->int:
    PADDIN = 'post'
    if data_type != 'Abstract':
        MAXLEN = 30
    else:
        MAXLEN = 250
    #loading the tokenizer 
    with open(f'./{tag_type}/tokinizer/{data_type}_tokenizer.json') as f:
        data = json.load(f)
        tokenizer = keras.preprocessing.text.tokenizer_from_json(data)
    
    tfidf = pickle.load(open(f'./{tag_type}/tfidf/{data_type}.pickle','rb')) 
    
    classes = [0]*n_classes

    for i in models:
        name = type(i).__name__
        if name in ('MultinomialNB' , 'LogisticRegression' , 'LinearSVC'):
            newtext = tfidf.transform(text)
            x = i.predict(newtext)[0]
            classes[x]+=1
        else:
            tokinized = tokenizer.texts_to_sequences(text)
            newtext = pad_sequences(tokinized,padding=PADDIN, maxlen=MAXLEN)
            x = i.predict(newtext)
            x = x.argmax()
            classes[x]+=1
    return [classes.index(max(classes)),round((max(classes)/5),2)]

def solver(models,text,tag_type,data_type):
    if tag_type == 'Dimension':
        n_classes = len(dimension_tags)
    else:
        n_classes = len(wos_tags)
    
    encoder = pickle.load(open(f'./encoder/{tag_type}.pickle','rb'))
    x = predictor(models,text,n_classes,tag_type=tag_type,data_type=data_type)
    return encoder.inverse_transform([x[0]]),x[1]


def load_model(tag_type,data_type):
    nb = pickle.load(open(f'./{tag_type}/{data_type}/MultinomialNB.pickel','rb'))
    logi = pickle.load(open(f'./{tag_type}/{data_type}/LogisticRegression.pickel','rb'))
    svm = pickle.load(open(f'./{tag_type}/{data_type}/LinearSVC.pickel','rb'))
    ann = keras.models.load_model(f'./{tag_type}/{data_type}/ann.h5')
    cnn = keras.models.load_model(f'./{tag_type}/{data_type}/cnn.h5')
    return [nb,logi,svm,ann,cnn]




dimension_tags = pd.read_csv('./Dimension/tags/tags.csv')
wos_tags = pd.read_csv('./WOS/tags/tags.csv')


def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])


def main():
    #load models
    abstact_models = load_model(tag_type=tag_type,data_type='Abstract')
    authorkeyword_models= load_model(tag_type=tag_type,data_type='Authorkeyword')
    articletitle_models = load_model(tag_type=tag_type,data_type='ArticleTitle')
    #serve models
    abstract_class = solver(abstact_models,[abstract],tag_type=tag_type,data_type='Abstract')
    authorkeyword_class = solver(authorkeyword_models,[keyword],tag_type=tag_type,data_type='Authorkeyword')
    articletitle_class = solver(articletitle_models,[title],tag_type=tag_type,data_type='ArticleTitle')
    #predict
    decision_dict = {
        'Article Title':{
            'Prediction': articletitle_class[0][0],
            'Confidence': f'{articletitle_class[1]*100}%'
            },
        'Author Keywords':{
            'Prediction': authorkeyword_class[0][0],
            'Confidence': f'{authorkeyword_class[1]*100}%'
            },
        'Abstract': {
            'Prediction': abstract_class[0][0],
            'Confidence': f'{abstract_class[1]*100}%'
            }
    }

    with column2:
        d = pd.DataFrame.from_dict(decision_dict)
        st.dataframe(d)
        count = Counter(d.loc['Prediction'].values)
        count = invert_dict(count)
        classname = count[max(count)]
        source = pd.read_csv('./Source/all.csv')
        source = source[source['classname']== classname]
        sourcelist = source['Source'].tolist()
        st.subheader("Common Publishing sources are : ")
        for i in sourcelist:
            st.markdown("- " + i)


with column1:
    tag_type = st.radio("Which type of tags you are Interested:",('WOS','Dimension'))
    if len(title) == 0 or len(keyword) == 0 or len(abstract) == 0:
        st.button('Compute', on_click=main, disabled= True)
    else:
        st.button('Compute', on_click=main, disabled= False)

#side bar

    if tag_type == 'Dimension':
        st.sidebar.title("Supported Topic")
        for index,value in enumerate(dimension_tags['classname'].tolist()):
            st.sidebar.markdown(f"{index+1}. {value}")
    elif tag_type == 'WOS':
        st.sidebar.title("Supported Topics")
        for index,value in enumerate(wos_tags['classname'].tolist()):
            st.sidebar.markdown(f"{index+1}. {value}")
            
