<h1>HMM-POS-Tagger</h1>


This is my attempt at a POS tagger using a Hidden Markov Model. Usage would be the following:

<p> >>> from nltk.corpus import treebank </p>
<p> >>> a = HMM.Model(treebank).train() </p>
<p> >>> a.tagger('Where are the boys?')</p>

['ADV', 'VERB', 'DET, 'NOUN', '.']

<p>If you are using pip you can install the required packages with the following command:</p>
<p>$ pip install -r requirements.txt</p>

