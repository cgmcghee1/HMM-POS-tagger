import nltk
from nltk.util import bigrams
import numpy as np
from nltk.tokenize import word_tokenize

class Model:

    def __init__(self, corpus):
        #takes input from nltk.corpus, for example treebank or brown

        self.words_tags = []

        for sent in corpus.tagged_sents():
            self.words_tags.append(("START", "START"))
            self.words_tags.extend([tup for tup in sent])
            self.words_tags.append(("END", "END"))

        self.tags = [tags for (words,tags) in self.words_tags]
        self.uniquetags = set(self.tags)
        self.uniquewords = set([words for (words,tags) in self.words_tags])
        self.tagbigrams = list(bigrams(self.tags))

    def train(self):


        self.tagdict = {tag:num for (num, tag) in enumerate(self.uniquetags)}
        self.numdict = {num:tag for (tag, num) in self.tagdict.items()}

        self.transition_probs = np.zeros(shape = (len(self.tagdict), len(self.tagdict))\
                                    , dtype = float)

        for tag_1 in self.tagdict:
            b = [[t_1] for [t_1,t] in self.tagbigrams if t_1 == tag_1]
            print(str(round(self.tagdict[tag_1]/len(self.uniquetags)*100, 1)) + '%')
            for tag in self.tagdict:
                a = [[t_1, t] for [t_1,t] in self.tagbigrams if t_1 == tag_1\
                         and t == tag]

                self.transition_probs[self.tagdict[tag]][self.tagdict[tag_1]]\
                    = len(a)/len(b)

        return(self)


    def emission_probs(self, w, t):
        count_w_t = len([[word, tag] for [word, tag] in self.words_tags if tag == t\
                         and word == w]) + 1
        count_t = len([[tag] for [word, tag] in self.words_tags if tag == t])
        emission_prob = count_w_t/count_t
        return(emission_prob)


    def tagger(self, sentence):
        splitsent = word_tokenize(sentence)
        self.vitmatrix = np.zeros(shape = (len(self.tagdict), len(splitsent)), dtype = float)

        backpointer = []

        for tag in self.tagdict:
             tag_1 = "START"
             self.vitmatrix[self.tagdict[tag]][0] = self.transition_probs\
                 [self.tagdict[tag]][self.tagdict[tag_1]] * self.emission_probs\
                 (splitsent[0],tag)

        index = np.argmax(self.vitmatrix[:,0])
        backpointer.append(self.numdict[index])

        for i in range(1, len(splitsent)):
            for tag in self.tagdict:
                maxarray = []
                for tag_1 in self.tagdict:
                    maxarray.append(self.vitmatrix[self.tagdict[tag_1]][i-1]*self.\
                                    transition_probs[self.tagdict[tag]][self.tagdict[tag_1]])

                vitmax = max(maxarray)
                self.vitmatrix[self.tagdict[tag]][i] = vitmax * self.emission_probs\
                    (splitsent[i],tag)

            index = np.argmax(self.vitmatrix[:,i])
            backpointer.append(self.numdict[index])

        return(backpointer)

