#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script trains a classifier to distinguish between the genders of authors.
The classifier is trained on original Hebrew data and tested on the same data using 5-fold cross validation. It is also tested on translation data and the results are saved to a file specified when calling the program.

Fetures used:
- relative tag counts
- relative tense counts (normalized over all tokens that have tense)
- relative binyan counts (normalizes over all tokens that have binyan)
- relative suffunction counts (normalized over all tokens)
- relative pos-trigram counts
- use of prefconj (relative count)
- use of relativizer (relative count)
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
    allTags = set() # normalize over total number of tokens
    allTenses = set() # normalize over all tokens that have tense
    allBinyan = set() # normalize over all tokens that have binyan
    allSuffunction = set() # normalize over total number of tokens
    allPos_trigrams = set() # normalize over total number of tokens - 2

    femSents = [] # List of lists (sentences, tokens)
    femToks = 0
    maleSents = []
    maleToks = 0

    # iterate over instances
    for inst in data:
        attr = inst.attrib

        # ignore instances that are verse
        if attr["verses"] == "Yes":
            continue

        # check gender
        author_gender = attr["author__gender"]
        
        # iterate over sentences
        for sent in inst:
            currTrigram = []
            text = sent.text
            lines = text.split("\n")[1:-1]

            if author_gender == "Female":
                femSents.append(lines)
            else:
                maleSents.append(lines)

            # iterate over words
            for line in lines:
                fields = line.strip().split("\t")
                tag = fields[4]
                tense = fields[14]
                biny = fields[15]
                suff = fields[23]

                currTrigram.append(tag)
                if len(currTrigram) > 3:
                    currTrigram = currTrigram[1:]
                    
                allTags.add(tag)

                # only consider tense if not NULL
                if tense != "NULL":
                    allTenses.add(tense)

                # only consider binyan if not NULL
                if biny != "NULL":
                    allBinyan.add(biny)

                allSuffunction.add(suff)

                if len(currTrigram) == 3:
                    trigram = tuple(currTrigram)
                    allPos_trigrams.add(trigram)

                # increase gender token count
                if author_gender == "Female":
                    femToks += 1
                else:
                    maleToks += 1

    # build feature vectors
    tagFeat = list(allTags)
    tenseFeat = list(allTenses)
    binyanFeat = list(allBinyan)
    suffFeat = list(allSuffunction)
    posFeat = list(allPos_trigrams)

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

        if tokCount >= 2000:
            femInst.append(inst[:])
            inst = []
            tokCount = 0
    femInst.append(inst[:])
    
    inst = []
    tokCount = 0
    for sent in maleSents:
        inst.append(sent)
        tokCount += len(sent)

        if tokCount >= 2000:
            maleInst.append(inst[:])
            inst = []
            tokCount = 0

            if len(maleInst) == len(femInst):
                break
            
    print("Number of fem instances: " + str(len(femInst)))
    print("Number of male instances: " + str(len(maleInst)))

    # generate feature vectors
    vectors_labels = []
    for sentences in femInst:
        feat = buildFeatures(sentences, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat)
        vectors_labels.append((feat, 0)) # Female class is '0'
    for sentences in maleInst:
        feat = buildFeatures(sentences, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat)
        vectors_labels.append((feat, 1)) # Male class is '1'

    # shuffle training instances
    np.random.shuffle(vectors_labels)

    vectors = [vec for (vec, label) in vectors_labels]
    labels = [label for (vec, label) in vectors_labels]
    
    return (vectors, labels, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat)



def buildTestFeatures(data, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat):

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

            if tokCount >= 2000:
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

            if tokCount >= 2000:
                maleInst[trans_gender].append(inst[:])
                inst = []
                tokCount = 0
        maleInst[trans_gender].append(inst[:])

    # generate feature vectors
    features = []
    for trans_gender in femInst:
        for sentences in femInst[trans_gender]:
            feat = buildFeatures(sentences, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat)
            features.append(feat)
    for trans_gender in maleInst:
        for sentences in maleInst[trans_gender]:
            feat = buildFeatures(sentences, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat)
            features.append(feat)

    

    genders = []
    for trans_gender in femInst:
        for i in range(len(femInst[trans_gender])):
            genders.append(("female", trans_gender))
    for trans_gender in maleInst:
        for i in range(len(maleInst[trans_gender])):
            genders.append(("male", trans_gender))


    return (features, genders)

def buildFeatures(sentences, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat):
    # Features to be considered
    tags = defaultdict(float) # counts for all instances
    tenses = defaultdict(float)
    binyan = defaultdict(float)
    prefconj = 0.0
    relativizer = 0.0
    suffunction = defaultdict(float)
    pos_trigrams = defaultdict(float)

    tokCount = 0
    tenseCount = 0
    binyanCount = 0
    trigramCount = 0

    # iterate over sentences
    for sentence in sentences:
        currTrigram = []

        # iterate over words
        for line in sentence:
            fields = line.strip().split("\t")
            tag = fields[4]
            tense = fields[14]
            biny = fields[15]
            prefc = fields[16]
            rel = fields[20]
            suff = fields[23]

            currTrigram.append(tag)
            if len(currTrigram) > 3:
                currTrigram = currTrigram[1:]

            tokCount += 1 # increment token count

            tags[tag] += 1

            # only consider tense if not NULL
            if tense != "NULL":
                tenses[tense] += 1
                tenseCount += 1

            # only consider binyan if not NULL
            if biny != "NULL":
                binyan[biny] += 1
                binyanCount += 1

            if prefc != "NULL":
                prefconj += 1

            if rel != "NULL":
                relativizer += 1

            suffunction[suff] += 1

            if len(currTrigram) == 3:
                trigram = tuple(currTrigram)
                pos_trigrams[trigram] += 1
                trigramCount += 1

    # relativize counts for instance
    for tag in tags:
        tags[tag] /= tokCount
    for tense in tenses:
        tenses[tense] /= tenseCount
    for biny in binyan:
        binyan[biny] /= binyanCount
    prefconj /= tokCount
    relativizer /= tokCount
    for suff in suffunction:
        suffunction[suff] /= tokCount
    for trigram in pos_trigrams:
        pos_trigrams[trigram] /= trigramCount

    featVec = []
    for tag in tagFeat:
        featVec.append(tags[tag])
    for tense in tenseFeat:
        featVec.append(tenses[tense])
    for biny in binyanFeat:
        featVec.append(binyan[biny])
    for suff in suffFeat:
        featVec.append(suffunction[suff])
    for trigram in posFeat:
        featVec.append(pos_trigrams[trigram])
    featVec.append(prefconj)
    featVec.append(relativizer)

    return (featVec)

    

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

    
    (vectors, genders, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat) = buildTrainingFeatures(originals)

    clf = trainClassifier(vectors, genders)
    (testVectors, testGenders) = buildTestFeatures(translations, tagFeat, tenseFeat, binyanFeat, suffFeat, posFeat)
    predictions = clf.predict(testVectors)

    with codecs.open(resultsfile, 'w', encoding="utf-8") as f:
        f.write("AUTHOR_GENDER\tTRANSLATOR_GENDER\tPREDICTED\n")
        for i in range(len(testGenders)):
            (author, trans) = testGenders[i]
            pred = predictions[i]
            f.write(author+"\t"+trans+"\t"+str(pred)+"\n")



    
    
