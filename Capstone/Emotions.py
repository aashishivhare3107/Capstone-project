#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


# In[14]:


emotion = pd.read_csv('text_emotion.csv')
print(emotion.shape)


# In[15]:


emotion.head()


# In[16]:


emotion.isnull().any()


# In[21]:


# checking out the negative comments from the emotion set

emotion[emotion['sentiment'] == 'hate'].head(10)


# In[20]:


# checking out the postive comments from the emotion set 

emotion[emotion['sentiment'] == 'love'].head(10)


# In[22]:


emotion['sentiment'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))


# In[24]:


# checking the distribution of tweet in the data

length_emotion = emotion['sentiment'].str.len().plot.hist(color = 'pink', figsize = (6, 4))


# In[25]:


# adding a column to represent the length of the tweet

emotion['len'] = emotion['sentiment'].str.len()

emotion.head(10)


# In[26]:


emotion.groupby('sentiment').describe()


# //emotion.groupby('len').mean()['sentiment'].plot.hist(color = 'black', figsize = (6, 4),)//
# //plt.title('variation of length')//
# //plt.xlabel('Length')//
# //plt.show()//
# 

# In[32]:


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(emotion.sentiment)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 30")


# In[34]:


from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize = 22)


# In[40]:


normal_words =' '.join([text for text in emotion['sentiment'][emotion['sentiment'] == 'love']])

wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()


# In[42]:


negative_words =' '.join([text for text in emotion['sentiment'][emotion['sentiment'] == 'hate']])

wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()


# In[51]:


# collecting the hashtags

def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[54]:


import re
# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(emotion['content'][emotion['sentiment'] == 'love'])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(emotion['content'][emotion['sentiment'] == 'worry'])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[58]:



import nltk


# In[59]:


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[60]:


a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[ ]:




