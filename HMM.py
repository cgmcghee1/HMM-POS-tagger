import nltk
from nltk.util import bigrams
import numpy as np
from nltk.tokenize import word_tokenize

class Model:
    """
    A class to implement a Hidden Markov Model POS tagger as described in Jurafsky and Martin's
    'Speech and Language Processing'(p148-155). The draft of the 3rd edition can be found here:

    https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf

    """


    def __init__(self, corpus):
        """
        The init function takes the tagged sentences in an NLTK corpus (specified by the corpus
        parameter) and appends a START and END tag/word to each sentence.

        :classattribute words_tags: list of tuples (word,tag) from input corpus
        :classattribute tags: list of tags from self.words_tags
        :classattribute uniquetags: set of unique tags from self.tags
        :classattribute uniquewords: set of unique words from self.words_tags
        :classattribute tagbigrams: list of bigrams of tags from self.tags

        """
        self.words_tags = []

        for sent in corpus.tagged_sents(tagset = 'universal'):
            self.words_tags.append(("START", "START"))
            self.words_tags.extend([tup for tup in sent])
            self.words_tags.append(("END", "END"))

        self.tags = [tags for (words,tags) in self.words_tags]
        self.uniquetags = set(self.tags)
        self.uniquewords = set([words for (words,tags) in self.words_tags])
        self.tagbigrams = list(bigrams(self.tags))

    def train(self):
        """
        Function which creates a numpy matrix of the transition probabilities between states (tags)

        :classattribute tagdict: a dictionary which maps an index to each unique tag
        :classattribute numdict: a dictionary which maps a tag to it's index from tagdict
        :classattribute transition_prosb: numpy matrix which stores the transition probabilities
        between states (tags)
        :classattribute tagbigrams: list of bigrams of tags from self.tags
        :classattribute uniquetags: set of unique tags from self.tags

        """

        self.tagdict = {tag:num for (num, tag) in enumerate(self.uniquetags)}
        self.numdict = {num:tag for (tag, num) in self.tagdict.items()}

        self.transition_probs = np.zeros(shape = (len(self.tagdict), len(self.tagdict))\
                                    , dtype = np.float32)

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
        """
        Function which returns the probability of a word being observed given a specific tag.

        :param w: input word
        :param t: input tag
        :classattribute words_tags: list of tuples (words, tags) from input corpus
        :return emission_prob: probability of a word being observed given a specific tag
        """

        count_w_t = len([[word, tag] for [word, tag] in self.words_tags if tag == t\
                         and word == w]) + 1
        count_t = len([[tag] for [word, tag] in self.words_tags if tag == t])
        emission_prob = count_w_t/count_t
        return(emission_prob)


    def tagger(self, sentence):
        """
        Function which implements the Viterbi algorithm to find the highest probability tag
        sequence.

        :param sentence: input string representing the sentence to be tagged
        :classattribute tagdict: a dictionary which maps an index to each unique tag
        :classattribute transition_probs: numpy matrix which stores the transition
        probabilities between states (tags)
        :classattribute numdict: a dictionary which maps a tag to it's index from tagdict
        :return bestpath: list of best tag sequence

        """

        splitsent = word_tokenize(sentence)
        vitmatrix = np.zeros(shape = (len(self.tagdict), len(splitsent)), dtype = np.float32)
        backpointer = np.zeros(shape = (len(self.tagdict), len(splitsent)), dtype = np.int)

        for tag in self.tagdict:
             tag_1 = "START"
             vitmatrix[self.tagdict[tag]][0] = self.transition_probs\
                 [self.tagdict[tag]][self.tagdict[tag_1]] * self.emission_probs\
                 (splitsent[0],tag)
             backpointer[self.tagdict[tag]][0] = 0

        for i in range(1, len(splitsent)):
            for tag in self.tagdict:
                max_array = []
                for tag_1 in self.tagdict:
                    max_array.append(vitmatrix[self.tagdict[tag_1]][i-1] * \
                                     self.transition_probs[self.tagdict[tag]][self.tagdict[tag_1]]
                                     * self.emission_probs(splitsent[i], tag))
                vitmatrix[self.tagdict[tag]][i] = np.max(max_array)
                backpointer[self.tagdict[tag]][i] = np.argmax(max_array)

        bestpathpointer = np.argmax(vitmatrix[:,-1])
        bestpath = []
        bestpath.append(self.numdict[bestpathpointer])

        for i in reversed(range(1, len(splitsent))):
            bestpathpointer = backpointer[bestpathpointer][i]
            bestpath.append(self.numdict[bestpathpointer])

        bestpath.reverse()
        return bestpath
