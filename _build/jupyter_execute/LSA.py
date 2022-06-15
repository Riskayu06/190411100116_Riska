#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet
# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# # LOADING THE DATASET

# In[2]:


df=pd.read_csv('techno.csv')


# In[3]:


df.head()


# In[4]:


# drop the nama.
df.drop(['nama'],axis=1,inplace=True)


# In[5]:


# drop the tanggal.
df.drop(['tanggal'],axis=1,inplace=True)


# In[6]:


# drop the kategori.
df.drop(['kategori'],axis=1,inplace=True)


# In[7]:


df.head(10)


# # DATA CLEANING & PRE-PROCESSING

# In[8]:


def clean_text(deskripsi):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(deskripsi)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[9]:


# time taking
df['deskripsi_text']=df['deskripsi'].apply(clean_text)


# In[10]:


df.head()


# In[11]:


df.drop(['deskripsi_text'],axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df['deskripsi'][0]


# # EXTRACTING THE FEATURES AND CREATING THE DOCUMENT-TERM-MATRIX ( DTM )

# In[14]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)


# In[15]:


vect_text=vect.fit_transform(df['deskripsi'])


# In[16]:


print(vect_text.shape)
print(vect_text)


# In[17]:


idf=vect.idf_


# In[19]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['setelah'])
print(dd['acara'])  # police is most common and forecast is least common among the news headlines.


# # TOPIC MODELLING Latent Semantic Analysis (LSA)

# In[20]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[21]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[22]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[23]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# In[24]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[25]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[26]:


lda_top=lda_model.fit_transform(vect_text)


# In[27]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top)


# In[28]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)


# In[29]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# In[30]:


print(lda_model.components_)
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# In[31]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

