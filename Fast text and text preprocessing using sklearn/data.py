#
# Created by Umanga Bista.
#

import glob
import os
import pandas as pd
from collections import namedtuple
from io import open

## A document class with following attributes
## id: document id
## category: category of document
## text: body of documment
Doc = namedtuple('Doc', 'id category text')

def read_doc(doc_path, encoding):
    '''
        reads a document from path
        input:
            - doc_path : path of document
            - encoding: encoding
        output: =>
            - doc: instance of Doc namedtuple
    '''
    category, _id = tuple(doc_path.split(os.path.sep)[-2:])
    fp = open(doc_path, 'r', encoding = encoding)
    text = fp.read()
    fp.close()
    return Doc(id = _id, category = category, text = text)

def read_dataset(path, encoding = "ISO-8859-1"):
    '''
        reads multiple documents from path
        input:
            - doc_path : path of document
            - encoding: encoding
        output: =>
            - docs: instances of Doc namedtuple returned as generator
    '''
    for doc_path in glob.glob(path + (os.path.sep + '*') * 2):
        yield read_doc(doc_path, encoding = encoding)

def read_as_df(path, encoding = "ISO-8859-1"):
    '''
        reads multiple documents from path
        input:
            - doc_path : path of document
            - encoding: encoding
        output: =>
            - docs: dataframe equivalent of doc Namedtuple
    '''
    dataset = list(read_dataset(path, encoding))
    return pd.DataFrame(dataset, columns = Doc._fields)
