#!/usr/bin/env python3
#This exercise was completed as part of a Machine Learning course of mooc.fi platform.
#url: https://courses.mooc.fi/org/uh-cs/courses/data-analysis-with-python-2024-2025

from collections import Counter
import urllib.request
from lxml import etree
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import sys

alphabet="abcdefghijklmnopqrstuvwxyzäö-"
alphabet_set = set(alphabet)

# Returns a list of Finnish words
def load_finnish():
    finnish_url="https://www.cs.helsinki.fi/u/jttoivon/dap/data/kotus-sanalista_v1/kotus-sanalista_v1.xml"
    filename="/Users/denis/Library/Application Support/tmc/vscode/mooc-data-analysis-with-python-2024-2025/part06-e03_word_classification/src/kotus-sanalista_v1.xml"
    load_from_net=False
    if load_from_net:
        with urllib.request.urlopen(finnish_url) as data:
            lines=[]
            for line in data:
                lines.append(line.decode('utf-8'))
        doc="".join(lines)
    else:
        with open(filename, "rb") as data:
            doc=data.read()
    tree = etree.XML(doc)
    s_elements = tree.xpath('/kotus-sanalista/st/s')
    return list(map(lambda s: s.text, s_elements))

def load_english():
    with open("/Users/denis/Library/Application Support/tmc/vscode/mooc-data-analysis-with-python-2024-2025/part06-e03_word_classification/src/words", encoding="utf-8") as data:
        lines=map(lambda s: s.rstrip(), data.readlines())
    return lines

def get_features(a):
    a_dupl = list(a)
    alphabet = "abcdefghijklmnopqrstuvwxyzäö-"
    data = np.zeros((len(a_dupl), len(alphabet)), dtype=int)

    for row, word in enumerate(a_dupl):
        for column, letter in enumerate(alphabet):
            word = word.lower()
            data[row, column] = word.count(letter)

    return data



def contains_valid_chars(s):
    alphabet = "-abcdefghijklmnopqrstuvwxyzäö"
    x = [1 if char in alphabet else 0 for char in s ]
    return sum(x) == len(x)

def get_features_and_labels():
    alphabet = "-abcdefghijklmnopqrstuvwxyzäö"
    fin = load_finnish()
    eng = load_english()

    fin = list([word for word in map(lambda x: x.lower(), fin) if contains_valid_chars(word)])
    eng = [word for word in eng if not word[0].isupper()]
    eng = list([word for word in map(lambda x: x.lower(), eng) if contains_valid_chars(word)])

    X = np.array(get_features(fin + eng))
    y = np.array([0 for i in range(len(fin))] + [1 for i in range(len(eng))])

    return (X, y)


def word_classification(words):
    X, y = get_features_and_labels()
    model = MultinomialNB()
    model.fit(X, y)

    words = get_features(words)
    result = model.predict(words)

    return result


def main():
    #print("Accuracy scores are:", word_classification())
    words = eval(sys.argv[1])
    print(word_classification(words))


if __name__ == "__main__":
    main()
