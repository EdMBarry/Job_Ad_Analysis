# Helper functions for CV processing

import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import string
import fitz
import pandas as pd
from collections import Counter
import en_core_web_sm
from spacy.matcher import PhraseMatcher

nlp = en_core_web_sm.load()


def get_files(my_path):
    """Takes a path to a directory and returns a list of files in that directory."""

    file_list = [os.path.join(my_path, f) for f in os.listdir(my_path) \
                if os.path.isfile(os.path.join(my_path, f))]
    return file_list


def pdf_extract(file):
    """ Extracts text from a PDF file. They need to be saved as such
        because they won't be available on job sites forever. 
        
        Arguments:
        file -- string of path to a PDF file.
        
        Returns:
        text -- the cleaned contents of the PDF document as a string.
        """
    
    
    doc = fitz.open(file)
    
    text = ''
    
    for page in doc:
        words = page.getText("text")

        punc = ["\n", "•", "(", ")", "\"", "'s", ";", ",", ".", '!']
        for pchar in punc:
            words = words.replace(pchar, "")
            words = words.lower()
        text = text + words
        
    return text


def candidate_profile(file, kw_list):
    """ Takes the given file and keyword dictionary and returns a database
        of the frequency of given domains. This can tell us if the profile
        is of a generalist or a specialist."""
    
    text = pdf_extract(file)
    kw_dataframe = pd.read_csv(kw_list)
    
    # Create a list of words belonging to each category
    stat_words = [nlp(text) for text in kw_dataframe['statistics'].dropna(axis=0)]
    ml_words   = [nlp(text) for text in kw_dataframe['machine_learning'].dropna(axis=0)]
    dl_words = [nlp(text) for text in kw_dataframe['deep_learning'].dropna(axis=0)]
    r_words = [nlp(text) for text in kw_dataframe['rstats'].dropna(axis=0)]
    py_words = [nlp(text) for text in kw_dataframe['python'].dropna(axis=0)]
    data_eng_words = [nlp(text) for text in kw_dataframe['data_engineering'].dropna(axis=0)]
    data_analysis_words = [nlp(text) for text in kw_dataframe['data_analysis'].dropna(axis=0)]
    
    # Match the words to the text
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("statistics", None, *stat_words)
    matcher.add("machine_learning", None, *ml_words)
    matcher.add("deep_learning", None, *dl_words)
    matcher.add("rstats", None, *r_words)
    matcher.add("python", None, *py_words)
    matcher.add("data_engineering", None, *data_eng_words)
    matcher.add("data_analysis", None, *data_analysis_words)
    doc = nlp(text)
    
    rules = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        rules.append((rule_id, span.text))

    keywords = "\n".join(f"{i[0]} {i[1]} ({j})" for i, j in Counter(rules).items())
    

    
    # Convert the strong of keywords to a dataframe
    df = pd.read_csv(StringIO(keywords), names = ["Keywords_List"])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ', 1).tolist(),
                       columns=["Domain", "Buzzword"])
    df2 = pd.DataFrame(df1.Buzzword.str.split('(', 1).tolist(),
                       columns = ['Buzzword', 'Count'])    

    df3 = pd.concat([df1['Domain'],df2['Buzzword'], df2['Count']], axis = 1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
    filename = filename.lower()
    name = pd.read_csv(StringIO(filename), names=['Company'])
    dataframe = pd.concat([name['Company'], df3], axis=1)
    dataframe['Company'].fillna(dataframe['Company'].iloc[0], inplace=True)
    
    return dataframe