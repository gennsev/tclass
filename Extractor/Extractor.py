import spacy
import numpy as np 
import pandas as pd
from spacy.pipeline import EntityRecognizer


class Extractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def NER(self, text):
        """
        Returns a list of entities found on the text.
        """
        lst=[]
        doc = self.nlp(text) 
        for ent in doc.ents:
            lst.append(ent.label_)
        return lst

    def POS(self, text):
        """
        Returns a list of tokens POS from the text.
        """
        lst=[]
        doc = self.nlp(text)
        for token in doc:
            lst.append(token.pos_)
        return lst

