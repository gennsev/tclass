#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import numpy as np 
import pandas as pd

nlp = spacy.load("en_core_web_md")


class POS_retriever:
    def fit(self, doc):
        pass

    def POS(self, data):
        lst=[]
        doc = nlp(data)
        for token in doc:
            lst.append(token.pos_)
        return ''.join(lst)

