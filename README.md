<h1>HMM-POS-Tagger</h1>


This is my attempt at a POS tagger using a Hidden Markov Model. Usage would be the following:

<p> >>> from nltk.corpus import brown </p>
<p> >>> a = HMM.Model(brown).train() </p>
<p> >>> a.tagger('Can you see the train?')</p>

['VERB', 'PRON', 'VERB', 'DET', 'NOUN', '.']



