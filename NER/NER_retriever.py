#!/usr/bin/env python
# coding: utf-8

# In[3]:


import spacy
import numpy as np 
import pandas as pd
from spacy.pipeline import EntityRecognizer

nlp = spacy.load("en_core_web_md")

class NER_retriever:
    def fit(self, doc):
        pass

    def NER(self, text):
        lst=[]
        doc = nlp(text) 
        for ent in doc.ents:
            lst.append(ent.label_)
        return lst


# In[ ]:




