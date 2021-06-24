#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the sgz_pke module."""

import os
import sys
import json
import logging
import xml.etree.ElementTree as etree
import spacy

from sgz_pke.data_structures import Document
from collections import Counter

class Reader(object):
    def read(self, path):
        raise NotImplementedError


class MinimalCoreNLPReader(Reader):
    """Minimal CoreNLP XML Parser."""

    def __init__(self):
        self.parser = etree.XMLParser()

    def read(self, path, **kwargs):
        sentences = []
        tree = etree.parse(path, self.parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in
                      sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in
                    sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            sentences.append({
                "words": [u.text for u in
                          sentence.iterfind("tokens/token/word")],
                "lemmas": [u.text for u in
                           sentence.iterfind("tokens/token/lemma")],
                "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets": [(starts[k], ends[k]) for k in
                                 range(len(starts))]
            })
            sentences[-1].update(sentence.attrib)

        doc = Document.from_sentences(sentences, input_file=path, **kwargs)

        return doc


# FIX
def fix_spacy_for_french(nlp):
    """Fixes https://github.com/boudinfl/pke/issues/115.
    For some special tokenisation cases, spacy do not assign a `pos` field.

    Taken from https://github.com/explosion/spaCy/issues/5179.
    """
    from spacy.symbols import TAG
    if nlp.lang != 'fr':
        # Only fix french model
        return nlp
    if '' not in [t.pos_ for t in nlp('est-ce')]:
        # If the bug does not happen do nothing
        return nlp
    rules = nlp.Defaults.tokenizer_exceptions

    for orth, token_dicts in rules.items():
        for token_dict in token_dicts:
            if TAG in token_dict:
                del token_dict[TAG]
    try:
        nlp.tokenizer = nlp.Defaults.create_tokenizer(nlp)  # this property assignment flushes the cache
    except Exception as e:
        # There was a problem fallback on using `pos = token.pos_ or token.tag_`
        ()
    return nlp


def list_linked_spacy_models():
    """ Read SPACY/data and return a list of link_name """
    spacy_data = os.path.join(spacy.info(silent=True)['Location'], 'data')
    linked = [d for d in os.listdir(spacy_data) if os.path.islink(os.path.join(spacy_data, d))]
    # linked = [os.path.join(spacy_data, d) for d in os.listdir(spacy_data)]
    # linked = {os.readlink(d): os.path.basename(d) for d in linked if os.path.islink(d)}
    return linked


def list_downloaded_spacy_models():
    """ Scan PYTHONPATH to find spacy models """
    models = []
    # For each directory in PYTHONPATH
    paths = [p for p in sys.path if os.path.isdir(p)]
    for site_package_dir in paths:
        # For each module
        modules = [os.path.join(site_package_dir, m) for m in os.listdir(site_package_dir)]
        modules = [m for m in modules if os.path.isdir(m)]
        for module_dir in modules:
            if 'meta.json' in os.listdir(module_dir):
                # Ensure the package we're in is a spacy model
                meta_path = os.path.join(module_dir, 'meta.json')
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('parent_package', '') == 'spacy':
                    models.append(module_dir)
    return models


def str2spacy(model):
    if int(spacy.__version__.split('.')[0]) < 3:
        downloaded_models = [os.path.basename(m) for m in list_downloaded_spacy_models()]
        links = list_linked_spacy_models()
    else:
        # As of spacy v3, links do not exist anymore and it is simpler to get a list of
        # downloaded models
        downloaded_models = list(spacy.info()['pipelines'])
        links = []
    filtered_downloaded = [m for m in downloaded_models if m[:2] == model]
    if model in downloaded_models + links:
        # Check whether `model` is the name of a model/link
        return model
    elif filtered_downloaded:
        # Check whether `model` is a lang code and corresponds to a downloaded model
        return filtered_downloaded[0]
    else:
        # Return asked model to have an informative error.
        return model


class RawTextReader(Reader):
    """Reader for raw text."""

    def __init__(self, language=None):
        """Constructor for RawTextReader.

        Args:
            language (str): language of text to process.
        """

        self.language = language

        if language is None:
            self.language = 'en'

        #self.spacy_doc = None
        self.ents_list = []
        self.ents_label_dict = {}
    
    def _get_name_entities(self,ents_full_list):
        self.ents_list.clear()
        self.ents_list = []
        self.ents_label_dict.clear()
        self.ents_label_dict = {}

        self.ents_list = list(set([X['text'] for X in ents_full_list]))
        #sort by num of words --> longest_sequence_selection() 处理时先考虑字数多的name entity
        self.ents_list.sort(key=lambda x: len(x.split()), reverse=True)
        #self.ents_label_dict = {}
        for w in self.ents_list:
            r_count = Counter([X['label_'] for X in ents_full_list if X['text']==w])
            self.ents_label_dict[w] = r_count.most_common()[0][0]
        #return ne,ne_label_dict

    def read(self, text, **kwargs):
        """Read the input file and use spacy to pre-process.

        Spacy model selection: By default this function will load the spacy
        model that is closest to the `language` parameter ('fr' language will
        load the spacy model linked to 'fr' or any 'fr_core_web_*' available
        model). In order to select the model that will be used please provide a
        preloaded model via the `spacy_model` parameter, or link the model you
        wish to use to the corresponding language code
        `python3 -m spacy link spacy_model lang_code`.

        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy (for spacy<3 compatibility, as of spacy v3 long texts
                should be splitted in smaller portions), default to
                1,000,000 characters (1mb).
            spacy_model (model): an already loaded spacy model.
        """

        spacy_model = kwargs.get('spacy_model', None)
        batch_size = kwargs.get('batch_size', -1) #-1,default no batch

        if spacy_model is None:
            spacy_kwargs = {'disable': ['ner', 'textcat', 'parser']}
            if 'max_length' in kwargs and kwargs['max_length']:
                spacy_kwargs['max_length'] = kwargs['max_length']

            try:
                spacy_model = spacy.load(str2spacy(self.language), **spacy_kwargs)
            except OSError:
                logging.warning('No spacy model for \'{}\' language.'.format(self.language))
                logging.warning('Falling back to using english model. There might '
                    'be tokenization and postagging errors. A list of available '
                    'spacy model is available at https://spacy.io/models.'.format(
                        self.language))
                spacy_model = spacy.load(str2spacy('en'), **spacy_kwargs)
            if int(spacy.__version__.split('.')[0]) < 3:
                sentencizer = spacy_model.create_pipe('sentencizer')
            else:
                sentencizer = 'sentencizer'
            spacy_model.add_pipe(sentencizer)

        spacy_model = fix_spacy_for_french(spacy_model)

        sentences = []
        ents_full_list = []

        if batch_size<=0:
            #"""原版本
            #'| Basic salary'-->'&#124; Basic salary'??
            spacy_doc = spacy_model(text)
            ents_full_list = [{'text':X.text,'label_':X.label_} for X in spacy_doc.ents]
            for sentence_id, sentence in enumerate(spacy_doc.sents):
                sentences.append({
                    "words": [token.text for token in sentence],
                    "lemmas": [token.lemma_ for token in sentence],
                    "dep": [token.dep_ for token in sentence],
                    # FIX : This is a fallback if `fix_spacy_for_french` does not work
                    "POS": [token.pos_ or token.tag_ for token in sentence],
                    "char_offsets": [(token.idx, token.idx + len(token.text))
                                        for token in sentence]
                })        
            #"""
        else:
            """
            采用nlp.pipe() 降低内存,但速度会降低
            https://spacy.io/usage/processing-pipelines#processing
            
            for doc in nlp.pipe(texts, batch_size=10000, n_threads=3):
            
            i int The index of the token within the parent document.        
            idx int The character offset of the token within the parent document.

            parsed_sentence = nlp(u'This is my sentence')
            [(token.text,token.i) for token in parsed_sentence]
                [(u'This', 0), (u'is', 1), (u'my', 2), (u'sentence', 3)]
            [(token.text,token.idx) for token in parsed_sentence]
                [(u'This', 0), (u'is', 5), (u'my', 8), (u'sentence', 11)]
            """

            #"""
            #第一步：分句
            other_pipes = [pipe for pipe in spacy_model.pipe_names if pipe not in \
                ['PySBDFactory','nltk_sentencizer','sentencizer']]
            disabled  = spacy_model.disable_pipes(*other_pipes)#只保留分句子pipeline
            #print(spacy_model.pipe_names)
            onlysent_doc = spacy_model(text)
            #print([sen.string for sen in onlysent_doc.sents])
            #print([token.is_sent_start for token in onlysent_doc])
            disabled.restore()
            #第二步：按句子doc化
            #other_pipes = [pipe for pipe in spacy_model.pipe_names if pipe in \
            #    ['PySBDFactory','nltk_sentencizer','sentencizer']]
            #disabled  = spacy_model.disable_pipes(*other_pipes)#去掉分句子pipeline
            #print(spacy_model.pipe_names)
            sen_char_offsets = 0
            
            for sen_doc in spacy_model.pipe([sen.string for sen in onlysent_doc.sents],\
                                            batch_size=batch_size):
                #print([sen.string for sen in sen_doc.sents])
                ents_full_list += [{'text':X.text,'label_':X.label_} for X in sen_doc.ents]
                for sentence_id, sentence in enumerate(sen_doc.sents):
                    char_offsets = \
                        [(token.idx+sen_char_offsets, token.idx+len(token.text)+sen_char_offsets) \
                            for token in sentence]
                    sentences.append({
                        "words": [token.text for token in sentence],
                        "lemmas": [token.lemma_ for token in sentence],
                        "dep": [token.dep_ for token in sentence],
                        # FIX : This is a fallback if `fix_spacy_for_french` does not work
                        "POS": [token.pos_ or token.tag_ for token in sentence],
                        "char_offsets": char_offsets
                    })
                sen_char_offsets = char_offsets[-1][1] + 1 #偏移按doc来计
            #"""
            
        self._get_name_entities(ents_full_list)
        """
        print()
        for s in sentences:
            print(s['words'],s['char_offsets'][0][0],s['char_offsets'][-1][1])
        print(len(sentences))
        exit()
        """
        doc = Document.from_sentences(sentences,
                                      input_file=kwargs.get('input_file', None),
                                      **kwargs)

        return doc

