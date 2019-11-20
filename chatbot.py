# Importing Necessary Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import wordnet # for lemmatization
from nltk import pos_tag # for parts of speech
from sklearn.feature_extraction.text import CountVectorizer # for word vectorization
from nltk.corpus import stopwords # for removing stop words
from sklearn.feature_extraction.text import TfidfVectorizer # to use tf-idf method
from sklearn.metrics import pairwise_distances # for cosine similarity

# getting excel file using pandas

df=pd.read_excel('dialog_talk_agent.xlsx')
df.head()


############### PREPROCESSING ################


# dropping the rows of all null values
df.dropna(axis=0,how='all',inplace=True)
df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)


# creating a loop to fill null values with previous responses
response=[]
for num,i in enumerate(df['Text Response']):
    if pd.isnull(i):
        response.append(response[num-2])
    else:
        response.append(i)


# or you can use fillna method to fill null values
'''df['response']=df['Text Response'].fillna(method='ffill')'''
df['response']=response
df.drop(columns='Text Response',inplace=True)
df.head()


############## LEMATIZATION ################


# function for lematization
def lematization(text):
    # step-1 : converting text to lower case
    text=str(text).lower() 
    # step-2 : removing special characters from the text
    stext=re.sub(r'[^ a-z]','',text)
    # step-3: tokenization
    tokens=nltk.word_tokenize(stext)
    # step-4 : lemmatization
    lema=wordnet.WordNetLemmatizer()
    tag_list=pos_tag(tokens,tagset=None) # for parts of speech
    lema_sent=[]    
    for token,pos_token in tag_list:
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)  # lemmatization
        lema_sent.append(lema_token)
    
    return " ".join(lema_sent)

# applying lematization function to the data

df['lema']=df['Context'].apply(lematization)
df.head()

'''
######### Removing Stop Words ############


# function to remove stop words
def stopw(text):
    # step - 1 : removing stop words    
    tag_list=pos_tag(nltk.word_tokenize(text),tagset=None)
    stop=stopwords.words('english')
    lema=wordnet.WordNetLemmatizer()
    lema_sent=[]
    for token,pos_token in tag_list:
        if token in stop:
            continue
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_sent.append(lema_token)
    return " ".join(lema_sent) 



# Applying stopwords function to the data

df['stop']=df['lema'].apply(stopw)
df.head()

############# creating all dataframes to csv files ##############

#df_tfidf.to_csv('TF-IDF.csv')
#df_bow.to_csv('BOW.csv')
#df.to_csv('Final.csv')


# creating BOW with count vectorizer method
cv=CountVectorizer()
x=cv.fit_transform(df['stop']).toarray()
df_bow=pd.DataFrame(x,columns=cv.get_feature_names())
df_bow.head()'''


########## creating TF-IDF Vectors ###########


# creating tf-idf vectors
tfidf=TfidfVectorizer()
x_tfidf=tfidf.fit_transform(df['lema']).toarray()
df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names())
df_tfidf.head()


############ Final Chat Function ################


# final function to convert query into tfidf vector and giving reply to that query by using cosine similarity method

def chat(query):
    lem=lematization(query)  # query lematization
    tf=tfidf.transform([lem]).toarray()  # transforming to tfidf vector
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine')  # checking cosine similarity with all vectors
    ind=cos.argmax() # getting maximum cosine similarity index
    
    '''bo=cv.transform([lem]).toarray()
    cos1=1-pairwise_distances(df_bow,bo,metric='cosine')
    ind1=cos1.argmax()'''
    
    return 'ro(BoT) :- '+df['response'].loc[ind] # df['response'].loc[ind1]

# final chatbot function
def chat_with_chatbot():    
    while True:
        inpu=str(input('ME : '))
        if inpu.lower()=='quit':
            print('(ro)BOT: Ok,Bye!!!!')
            break
        else:
            print(chat(inpu))
         



   
            




