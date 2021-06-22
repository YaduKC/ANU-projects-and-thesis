#
# Created by Umanga Bista.
#

import re
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

## Stemmer
stemmer = SnowballStemmer("english")
stopwords_en = set(stopwords.words('english'))

## now build a custom tokenizer based on these
__tokenization_pattern = r'''(?x)          # set flag to allow verbose regexps
        \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''

tokenizer = nltk.tokenize.regexp.RegexpTokenizer(__tokenization_pattern)


def preprocessor(text):
    '''
        turns text into tokens after tokenization, stemming, stop words removal
        imput:
            - text: document to process
        output: =>
            - tokens: list of tokens after tokenization, stemming, stop words removal
    '''
    stems = []
    tokens = tokenizer.tokenize(text.lower())

    for token in tokens:
        if token.isalpha() and token not in stopwords_en:
            stems.append(str(stemmer.stem(token)))

    return stems
