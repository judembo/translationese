#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script trains a classifier to distinguish between the genders of authors. This is used as a 'sanity check', so the only features used are the relative frequencies of lemmas, in this setup of 300 most frequent ones.
The classifier is trained on original Hebrew data and tested on the same data using 5-fold cross validation. It is also tested on translation data and the results are saved to a file specified when calling the program.
'''

import xml.etree.ElementTree as ET
import sys
from collections import defaultdict
import math
import codecs
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

def buildTrainingFeatures(data):
    # Features to be considered
    allTokens =  defaultdict(int)  # dictionary with token counts, so we can select the most frequent ones to be considered
    lemmaTags = defaultdict(int)
    
    femSents = [] # List of lists (sentences, tokens)
    maleSents = []

    # iterate over instances
    for inst in data:
        attr = inst.attrib

        # ignore instances that are verse
        if attr["verses"] == "Yes":
            continue

        # get gender
        author_gender = attr["author__gender"]
        
        # iterate over sentences
        for sent in inst:
            text = sent.text
            lines = text.split("\n")[1:-1]

            if author_gender == "Female":
                femSents.append(lines)
            elif author_gender =="Male":
                maleSents.append(lines)

            # iterate over words to get token counts
            for line in lines:
                fields = line.strip().split("\t")
                token = fields[2] # lemma
                tag = fields[4]
                allTokens[token] += 1
                lemmaTags[(token,tag)] += 1

    # build feature vectors
    token_counts = [(key, value) for (key, value) in allTokens.items()]
    token_counts_sorted = sorted(token_counts, key=lambda x: x[1], reverse=True)[:300]  # define the top-x words to be considered
    tokenFeat = [token for (token, count) in token_counts_sorted] # get a list of the tokens to be considered, from which the feature vectors can be built

    lemmaTagCounts = [(key, value) for (key, value) in lemmaTags.items()]
    lemmaCountsTop = sorted(lemmaTagCounts, key=lambda x: x[1], reverse=True)[:1000]

    # Print Token counts to afile
    with codecs.open("lemmaCounts.txt", 'w', encoding="utf-8") as f:
        for ((lemma, tag), count) in lemmaCountsTop:
            f.write(lemma + "\t" + tag + "\t" + str(count) + "\n")

            
    # shuffle sentences
    np.random.shuffle(femSents)
    np.random.shuffle(maleSents)


    # generate artificial instances
    femInst = []
    maleInst = []

    inst = []
    tokCount = 0
    for sent in femSents:
        inst.append(sent)
        tokCount += len(sent)

        if tokCount >= 2000:  # set number of tokens per instance
            femInst.append(inst[:])
            inst = []
            tokCount = 0
    femInst.append(inst[:])
    
    inst = []
    tokCount = 0
    for sent in maleSents:
        inst.append(sent)
        tokCount += len(sent)

        if tokCount >= 2000: # set number of tokens per instance
            maleInst.append(inst[:])
            inst = []
            tokCount = 0

            if len(maleInst) == len(femInst): # we want the same number of male and female instances
                break

    writeOrigInstances("origInstances.txt", femInst, maleInst)
    print(len(femInst))

    # generate feature vectors
    vectors_labels = []
    for sentences in femInst:
        feat = buildFeatures(sentences, tokenFeat)
        vectors_labels.append((feat, 0)) # Female class is '0'
    for sentences in maleInst:
        feat = buildFeatures(sentences, tokenFeat)
        vectors_labels.append((feat, 1)) # Male class is '1'

    # shuffle training instances, not sure if this is necessary?
    np.random.shuffle(vectors_labels)

    vectors = [vec for (vec, label) in vectors_labels]
    labels = [label for (vec, label) in vectors_labels]

    return (vectors, labels, tokenFeat)



def buildTestFeatures(data, tokenFeat):

    femSents = defaultdict(list) # dictionary for female authors, with one list (of lists sentences, tokens) per translator gender
    maleSents = defaultdict(list)

    # iterate over instances
    for inst in data:
        attr = inst.attrib

        # ignore instances that are verse
        if attr["verses"] == "Yes":
            continue

        # check gender
        author_gender = attr["author__gender"]
        trans_gender = attr["translator_s__gender"]

        # iterate over sentences
        for sent in inst:
            text = sent.text
            lines = text.split("\n")[1:-1]

            if author_gender == "Female":
                femSents[trans_gender].append(lines)
            elif author_gender == "Male":
                maleSents[trans_gender].append(lines)

    # shuffle sentences
    for trans_gender, sentences in femSents.items():
        np.random.shuffle(sentences)
    for trans_gender, sentences in maleSents.items():
        np.random.shuffle(sentences)

    # generate artificial instances
    femInst = defaultdict(list)
    maleInst = defaultdict(list)

    for trans_gender in femSents:
        inst = []
        tokCount = 0
        for sent in femSents[trans_gender]:
            inst.append(sent)
            tokCount += len(sent)

            if tokCount >= 2000: # set number of tokens per instance
                femInst[trans_gender].append(inst[:])
                inst = []
                tokCount = 0
        femInst[trans_gender].append(inst[:])

    for trans_gender in maleSents:
        inst = []
        tokCount = 0
        for sent in maleSents[trans_gender]:
            inst.append(sent)
            tokCount += len(sent)

            if tokCount >= 2000: # set number of tokens per instance
                maleInst[trans_gender].append(inst[:])
                inst = []
                tokCount = 0
        maleInst[trans_gender].append(inst[:])

    writeTransInstances("translationInstances.txt", femInst, maleInst)

    # generate feature vectors
    features = []
    genders = []
    for trans_gender in femInst:
        for sentences in femInst[trans_gender]:
            feat = buildFeatures(sentences, tokenFeat)
            features.append(feat)
            genders.append(("female", trans_gender))
    for trans_gender in maleInst:
        for sentences in maleInst[trans_gender]:
            feat = buildFeatures(sentences, tokenFeat)
            features.append(feat)
            genders.append(("male", trans_gender))

    return (features, genders)

def buildFeatures(sentences, tokenFeat):
    # Features to be considered
    tokens = defaultdict(float) # counts for all instances
    tokCount = 0
    
    # iterate over sentences
    for sentence in sentences:

        # iterate over words
        for line in sentence:
            fields = line.strip().split("\t")
            token = fields[2] # lemma

            tokCount += 1 # increment token count
            tokens[token] += 1

    # normalize counts
    for token in tokens:
        tokens[token] /= tokCount

    featVec = []
    for token in tokenFeat:
        featVec.append(tokens[token])

    return (featVec)

'''
Write the artificial 'original' instances containing the shuffled sentences to a file.
'''
def writeOrigInstances(fileName, femInst, maleInst):
    i = 0
    with codecs.open(fileName, 'w', encoding="utf-8") as f:
        for inst in femInst:
            i += 1
            f.write('<instance id="' + str(i) + '" author_gender="Female">\n')
            for sent in inst:
                text = ""
                for line in sent:
                    text += line.strip().split("\t")[0] + " "
                f.write(text + "\n")
            f.write("</instance>\n")
        for inst in maleInst:
            i += 1
            f.write('<instance id="' + str(i) + '" author_gender="Male">\n')
            for sent in inst:
                text = ""
                for line in sent:
                    text += line.strip().split("\t")[0] + " "
                f.write(text + "\n")
            f.write("</instance>\n")

'''
Write the artificial translation instances containing the shuffled sentences to a file.
'''
def writeTransInstances(fileName, femInst, maleInst):
    i = 0
    with codecs.open(fileName, 'w', encoding="utf-8") as f:
        for trans_gender in femInst:
            for inst in femInst[trans_gender]:
                i += 1
                f.write('<instance id="' + str(i) + '" author_gender="Female" translator_gender="' + trans_gender + '">\n')
                for sent in inst:
                    text = ""
                    for line in sent:
                        text += line.strip().split("\t")[0] + " "
                    f.write(text + "\n")
                f.write("</instance>\n")
        for trans_gender in maleInst:
            for inst in maleInst[trans_gender]:
                i += 1
                f.write('<instance id="' + str(i) + '" author_gender="Male" translator_gender="' + trans_gender + '">\n')
                for sent in inst:
                    text = ""
                    for line in sent:
                        text += line.strip().split("\t")[0] + " "
                    f.write(text + "\n")
                f.write("</instance>\n")
            

    
    

def trainClassifier(features, labels):
    clf = svm.SVC()
    scores = cross_val_score(clf, features, labels, cv=5, scoring='f1')
    print("F1")
    print(scores)
    print(sum(scores)/5)
    scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')
    print("accuracy")
    print(scores)
    print(sum(scores)/5)
    clf.fit(features, labels)
    return clf
    


def parse(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    originals = root[0]
    translations = root[1]

    return (originals, translations)

if __name__=='__main__':
    corpusfile = sys.argv[1]
    resultsfile = sys.argv[2]

    (originals, translations) = parse(corpusfile)

    
    (vectors, genders, tokenFeat) = buildTrainingFeatures(originals)

    clf = trainClassifier(vectors, genders)
    (testVectors, testGenders) = buildTestFeatures(translations, tokenFeat)
    predictions = clf.predict(testVectors)

    with codecs.open(resultsfile, 'w', encoding="utf-8") as f:
        f.write("AUTHOR_GENDER\tTRANSLATOR_GENDER\tPREDICTED\n")
        for i in range(len(testGenders)):
            (author, trans) = testGenders[i]
            pred = predictions[i]
            f.write(author+"\t"+trans+"\t"+str(pred)+"\n")



    
    
