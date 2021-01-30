<h1>HMM-POS-Tagger</h1>


This is my first attempt at a POS tagger using a Hidden Markov Model. Usage would be the following:

>>> from nltk.corpus import brown
>>> a = HMM.Model(brown).train()
>>> a.tagger('Can you see the train?')
['VERB', 'PRON', 'VERB', 'DET', 'NOUN', '.']



