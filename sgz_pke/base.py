# -*- coding: utf-8 -*-

"""Base classes for the sgz_pke module."""

""" fix: 修改length(影响shift/offsets),gap计算-->避免build_topic_graph()出错
1. self.length = len(words) ['Boris Johson','went','to','New York']  【DONE】
load_document-->read(): "words": [token.text for token in sentence]  spacy_doc.sents
-->from_sentences() s = Sentence(words=sentence['words'])-->self.length = len(words)

2. longest_sequence_selection: offset=shift+seq[0] 【DONE】
candidate_selection-->longest_sequence_selection: shift = sum([s.length for s in self.sentences[0:i]])
-->add_candidate: offset=shift+seq[0], self.candidates[lexical_form].offsets.append(offset)
   
only for MultipartiteRank TopicRank
3. build_topic_graph: gap  【】
candidate_weighting --> build_topic_graph: p_i in self.candidates[node_i].offsets, gap = abs(p_i - p_j)...weights.append(1.0 / gap)
"""

from collections import defaultdict

from sgz_pke.data_structures import Candidate, Document
from sgz_pke.readers import MinimalCoreNLPReader, RawTextReader

from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.tag.mapping import map_tag
from nltk.stem.snowball import SnowballStemmer, PorterStemmer

from .langcodes import LANGUAGE_CODE_BY_NAME

import string
from string import punctuation
import os
import logging
import codecs

from six import string_types

from builtins import str

import sgz_modules.constant as sgzconstant
from collections import Counter
import re

# The language management should be in `sgz_pke.utils` but it would create a circular import.

get_alpha_2 = lambda l: LANGUAGE_CODE_BY_NAME[l]

lang_stopwords = {get_alpha_2(l): l for l in stopwords._fileids}

lang_stem = {get_alpha_2(l): l for l in set(SnowballStemmer.languages) - set(['porter'])}
lang_stem.update({'en': 'porter'})

PRINT_NO_STEM_WARNING = defaultdict(lambda: True)
PRINT_NO_STWO_WARNING = defaultdict(lambda: True)


def get_stopwords(lang):
    """Provide stopwords for the given language, or default value.

    If stopwords are not available for a given language, a default value is
    returned and a warning is displayed
    :param lang: Alpha-2 language code.
    :type lang: str
    :returns: A list of stop words or an empty list.
    :rtype: {List}
    """
    global PRINT_NO_STWO_WARNING
    try:
        lang = lang_stopwords[lang]
        return stopwords.words(lang)
    except KeyError:
        if PRINT_NO_STWO_WARNING[lang]:
            logging.warning('No stopwords for \'{}\' language.'.format(lang))
            logging.warning(
                'Please provide custom stoplist if willing to use stopwords. Or '
                'update nltk\'s `stopwords` corpora using `nltk.download(\'stopwords\')`')
            PRINT_NO_STWO_WARNING[lang] = False
        return []


def get_stemmer_func(lang):
    """Provide steming function for the given language, or identity function.

    If stemming is not available for a given language, a default value is
    returned and a warning is displayed
    :param lang: Alpha-2 language code.
    :type lang: str
    :returns: A function to stem a word (or the identity function).
    :rtype: {Callable[[str], str]}
    """
    global PRINT_NO_STEM_WARNING
    try:
        lang = lang_stem[lang]
        ignore_sw = lang != 'porter'  # PorterStemmer do not use stop_words
        stemmer = SnowballStemmer(lang, ignore_stopwords=ignore_sw)
        return stemmer.stem
    except KeyError:
        if PRINT_NO_STEM_WARNING[lang]:
            logging.warning('No stemmer for \'{}\' language.'.format(lang))
            logging.warning('Stemming will not be applied.')
            PRINT_NO_STEM_WARNING[lang] = False
        return lambda x: x


escaped_punctuation = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']',
                       '-lcb-': '{', '-rcb-': '}'}


def is_file_path(input):
    try:
        return os.path.isfile(input)
    except Exception:
        # On some windows version the maximum path length is 255. When calling
        #  `os.path.isfile` on long string it will raise a ValueError.
        # We return false as even is the string is a file_path we won't be able
        #  to open it
        return False


def is_corenlp(input):
    return is_file_path(input) and input.endswith('.xml')


class LoadFile(object):
    """The LoadFile class that provides base functions."""

    def __init__(self):
        """Initializer for LoadFile class."""

        #两个或以上的letter重复
        self.repeat_letters_re = re.compile(r"^([a-zA-Z])\1{1,}$")

        self.ents_list = None
        self.ents_label_dict = None
        """list, save name entities of spacy nlp"""

        self.input_file = None
        """Path to the input file."""

        self.language = None
        """Language of the input file."""

        self.normalization = None
        """Word normalization method."""

        self.sentences = []
        """Sentence container (list of Sentence objects)."""

        self.candidates = defaultdict(Candidate)
        """Keyphrase candidates container (dict of Candidate objects)."""

        self.weights = {}
        """Weight container (can be either word or candidate weights)."""

        self._models = os.path.join(os.path.dirname(__file__), 'models')
        """Root path of the models."""

        self._df_counts = os.path.join(self._models, "df-semeval2010.tsv.gz")
        """Path to the document frequency counts provided in sgz_pke."""

        self.stoplist = None
        """List of stopwords."""

    def load_document(self, input, **kwargs):
        """Loads the content of a document/string/stream in a given language.

        Args:
            input (str): input.
            language (str): language of the input, defaults to 'en'.
            encoding (str): encoding of the raw file.
            normalization (str): word normalization method, defaults to
                'stemming'. Other possible values are 'lemmatization' or 'None'
                for using word surface forms instead of stems/lemmas.
        """

        # get the language parameter
        language = kwargs.get('language', 'en')

        # initialize document
        doc = Document()

        if is_corenlp(input):
            path = input
            parser = MinimalCoreNLPReader()
            doc = parser.read(path=input, **kwargs)
            doc.is_corenlp_file = True
        elif is_file_path(input):
            path = input
            with open(path, encoding=kwargs.get('encoding', 'utf-8')) as f:
                input = f.read()
            parser = RawTextReader(language=language)
            doc = parser.read(text=input, path=path, **kwargs)
            #self.ents_list,self.ents_label_dict = parser.get_name_entities()# called after parser.read()
            #parser.spacy_doc = None #手动清除
            self.ents_list = parser.ents_list
            self.ents_label_dict = parser.ents_label_dict
        elif isinstance(input, str):
            parser = RawTextReader(language=language)
            doc = parser.read(text=input, **kwargs)
            #self.ents_list,self.ents_label_dict = parser.get_name_entities()# called after parser.read()
            #parser.spacy_doc = None #手动清除
            self.ents_list = parser.ents_list
            self.ents_label_dict = parser.ents_label_dict
        else:
            logging.error('Cannot process input. It is neither a file path '
                          'or a string: {}'.format(type(input)))
            return
        
        # set the input file
        self.input_file = doc.input_file

        # set the language of the document
        self.language = language

        # set the sentences
        self.sentences = doc.sentences

        # initialize the stoplist
        self.stoplist = get_stopwords(self.language)

        # word normalization
        self.normalization = kwargs.get('normalization', 'stemming')

        if self.normalization == 'stemming':
            stem = get_stemmer_func(self.language)
            get_stem = lambda s: [stem(w).lower() for w in s.words]
        else:
            get_stem = lambda s: [w.lower() for w in s.words]

        # Populate Sentence.stems according to normalization
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = get_stem(sentence)

        # POS normalization
        if getattr(doc, 'is_corenlp_file', False):
            self.normalize_pos_tags()
            self.unescape_punctuation_marks()

    def normalize_pos_tags(self):
        """Normalizes the PoS tags from udp-penn to UD."""

        if self.language == 'en':
            # iterate throughout the sentences
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].pos = [map_tag('en-ptb', 'universal', tag)
                                         for tag in sentence.pos]

    def unescape_punctuation_marks(self):
        """Replaces the special punctuation marks produced by CoreNLP."""

        for i, sentence in enumerate(self.sentences):
            for j, word in enumerate(sentence.words):
                l_word = word.lower()
                self.sentences[i].words[j] = escaped_punctuation.get(l_word,
                                                                     word)

    def is_redundant(self, candidate, candidate_surface_form, prev, prev_surface_forms,
                    minimum_length=1):
        """Test if one candidate is redundant with respect to a list of already
        selected candidates. A candidate is considered redundant if it is
        included in another candidate that is ranked higher in the list.

        Args:
            candidate (str): the lexical form of the candidate.
            prev (list): the list of already selected candidates (lexical
                forms).
            minimum_length (int): minimum length (in words) of the candidate
                to be considered, defaults to 1.
        """
        # get the tokenized lexical form from the candidate
        candidate = self.candidates[candidate].lexical_form

        # only consider candidate greater than one word
        if len(candidate) < minimum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [self.candidates[u].lexical_form for u in prev]
        
        #print(prev,'<-->',candidate)
        #check surface form of selected candidates
        candidate_surface_form_words = candidate_surface_form.split()
        for w in prev_surface_forms:
            #if candidate_surface_form in w:
            w_words = w.split()
            if any(candidate_surface_form_words==w_words[i:i+len(candidate_surface_form_words)]
                    for i in range(len(w_words))):
                #print('              ',candidate_surface_form,'--candidate redundant-->',w)
                return True

        # loop through the already selected candidates
        for prev_candidate in prev:
            for i in range(len(prev_candidate) - len(candidate) + 1):
                if candidate == prev_candidate[i:i + len(candidate)]:
                    return True
        return False

    def get_n_best(self, n=10, redundancy_removal=False, stemming=False,
                surface_form_lowercase=True,
                del_repeat_letters=True,
                del_ne_labels=sgzconstant.DEL_NE_LABELS):
        """Returns the n-best candidates given the weights.

        Args:
            n (int): the number of candidates, defaults to 10.
            redundancy_removal (bool): whether redundant keyphrases are
                filtered out from the n-best list, defaults to False.
            stemming (bool): whether to extract stems or surface forms
                (lowercased, first occurring form of candidate), default to
                False.
        """
        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)
        #print()
        #print('get_n_best()---all candidates---',len(best))
        #print(best)
        #print(len(self.candidates.keys()))
        """self.weights
        {'inflat': 0.059646288645499375, 'us': 0.05018774158352419, 'equiti': 0.025886558913396324, 'good year': 0.02910111112823012}
        """
        best_surface_forms = []
        best_pos_dict = {}
        for candidate in best:
            r_count = Counter([' '.join(w) for w in self.candidates[candidate].surface_forms])
            #candidate_surface_form = r_count.most_common()[0][0]
            best_surface_forms.append(r_count.most_common()[0][0])#candidate_surface_form)
            r_count = Counter([' '.join(w) for w in self.candidates[candidate].pos_patterns])
            best_pos_dict[candidate] = r_count.most_common()[0][0]
        #print(best_surface_forms) 
        #print(best_pos)
        #print()

        #去掉单个字符的重复单词，如 xxxx, yyy, aa, bbbbb ^([a-zA-Z])\1{1,}$
        if del_repeat_letters:
            repeat_best = []
            repeat_best_surface_forms = []
            for idx,candidate in enumerate(best):
                if self.repeat_letters_re.match(best_surface_forms[idx]):
                    #print('         ----->',best_surface_forms[idx])
                    continue
                # add the candidate otherwise
                repeat_best.append(candidate)
                #get the words from the most occurring surface form
                repeat_best_surface_forms.append(best_surface_forms[idx])
            best = repeat_best
            best_surface_forms = repeat_best_surface_forms            
        
        #remove unwanted name entity labels
        if del_ne_labels:
            ne_best = []
            ne_best_surface_forms = []
            for idx,candidate in enumerate(best):
                #'first'-'ORDINAL'  'three','thousands'-'CARDINAL'
                #self.ents_list,self.ents_label_dict
                if best_surface_forms[idx] in self.ents_label_dict:
                    nelabel = self.ents_label_dict[best_surface_forms[idx]]
                    if nelabel in del_ne_labels:# == 'ORDINAL' or nelabel == 'CARDINAL':
                        #print('        --del-->',best_surface_forms[idx],nelabel)
                        continue
                # add the candidate otherwise
                ne_best.append(candidate)
                #get the words from the most occurring surface form
                ne_best_surface_forms.append(best_surface_forms[idx])
            best = ne_best
            best_surface_forms = ne_best_surface_forms

        # remove redundant candidates
        if redundancy_removal:
            # initialize a new container for non redundant candidates
            non_redundant_best = []
            non_redundant_best_surface_forms = []
            # loop through the best candidates
            for idx,candidate in enumerate(best):
                """
                #'first'-'ORDINAL'  'three','thousands'-'CARDINAL'
                #self.ents_list,self.ents_label_dict
                if best_surface_forms[idx] in self.ents_label_dict:
                    nelabel = self.ents_label_dict[best_surface_forms[idx]]
                    if nelabel == 'ORDINAL' or nelabel == 'CARDINAL':
                        print('        ',best_surface_forms[idx],nelabel)
                        continue
                """
                # test wether candidate is redundant
                if self.is_redundant(candidate,best_surface_forms[idx],
                                    non_redundant_best,non_redundant_best_surface_forms):
                    continue
                # Waters <--Replace-- Maxine Waters: (name entity)
                # time series <-- time series data
                replaced = False
                for ii,w in enumerate(non_redundant_best_surface_forms):
                    bsf = best_surface_forms[idx]
                    bsfl = bsf.split()
                    wl = w.split()
                    #if any(wl==bsfl[j:j+len(wl)] for j in range(len(bsfl))) and bsf in self.ents_list:
                    #if (bsf in self.ents_list and any(wl==bsfl[j:j+len(wl)] for j in range(len(bsfl)))) or \
                    #    (bsf not in self.ents_list and any(w.lower().split()==bsf.lower().split()[j:j+len(wl)] for j in range(len(bsfl)))):
                    if any(w.lower().split()==bsf.lower().split()[j:j+len(wl)] for j in range(len(bsfl))):
                        #print('        ',w,'--in-->',best_surface_forms[idx])
                        non_redundant_best[ii] = candidate
                        non_redundant_best_surface_forms[ii] = best_surface_forms[idx]
                        replaced = True
                if replaced:
                    #查重
                    temp = []
                    for nrb in non_redundant_best:
                        if nrb not in temp:
                            temp.append(nrb)
                    non_redundant_best = temp 
                    temp = []
                    for nrbs in non_redundant_best_surface_forms:
                        if nrbs not in temp:
                            temp.append(nrbs)
                    non_redundant_best_surface_forms = temp 
                    continue
                # add the candidate otherwise
                non_redundant_best.append(candidate)
                #get the words from the most occurring surface form
                non_redundant_best_surface_forms.append(best_surface_forms[idx])
                # break computation if the n-best are found
                if len(non_redundant_best) >= n:
                    break
            # copy non redundant candidates in best container
            best = non_redundant_best
            best_surface_forms = non_redundant_best_surface_forms

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        if not stemming:
            """原有代码,用的surface_forms[]第1个-->改为most frequent
            n_best = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                       self.weights[u]) for u in best[:min(n, len(best))]]
            """
            # only lower case
            #n_best = [(best_surface_forms[i].lower() if lowercase else best_surface_forms[i],
            #           self.weights[best[i]]) for i in range(min(n, len(best)))]
            n_best = [(best_surface_forms[i] if not surface_form_lowercase and 
                        (best_surface_forms[i] in self.ents_list or 
                        any(ent in best_surface_forms[i] for ent in self.ents_list) or 
                        'PROPN' in best_pos_dict[best[i]]) 
                        else best_surface_forms[i].lower(),
                       self.weights[best[i]]) for i in range(min(n, len(best)))]

        # return the list of best candidates
        return n_best

    def add_candidate(self, words, stems, pos, offset, sentence_id,candidate_char_offset=()):
        """Add a keyphrase candidate to the candidates container.

        Args:
            words (list): the words (surface form) of the candidate.
            stems (list): the stemmed words of the candidate.
            pos (list): the Part-Of-Speeches of the words in the candidate.
            offset (int): the offset of the first word of the candidate.
            sentence_id (int): the sentence id of the candidate.
        """

        # build the lexical (canonical) form of the candidate using stems
        lexical_form = ' '.join(stems)

        #记录candidate在句子里的char offset
        if len(candidate_char_offset)>0:
            self.candidates[lexical_form].candidate_char_offsets.append(candidate_char_offset)

        # add/update the surface forms
        self.candidates[lexical_form].surface_forms.append(words)

        # add/update the lexical_form
        self.candidates[lexical_form].lexical_form = stems

        # add/update the POS patterns
        self.candidates[lexical_form].pos_patterns.append(pos)

        # add/update the offsets
        self.candidates[lexical_form].offsets.append(offset)

        # add/update the sentence ids
        self.candidates[lexical_form].sentence_ids.append(sentence_id)

    def ngram_selection(self, n=3):
        """Select all the n-grams and populate the candidate container.

        Args:
            n (int): the n-gram length, defaults to 3.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # limit the maximum n for short sentence
            skip = min(n, sentence.length)

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # generate the ngrams
            for j in range(sentence.length):
                for k in range(j + 1, min(j + 1 + skip, sentence.length + 1)):
                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[j:k],
                                       stems=sentence.stems[j:k],
                                       pos=sentence.pos[j:k],
                                       offset=shift + j,
                                       sentence_id=i)

    def longest_pos_sequence_selection(self, valid_pos=None):
        self.longest_sequence_selection(
            key=lambda s: s.pos, valid_values=valid_pos)

    def longest_keyword_sequence_selection(self, keywords):
        self.longest_sequence_selection(
            key=lambda s: s.stems, valid_values=keywords)

    def longest_sequence_selection(self, key, valid_values):
        """Select the longest sequences of given POS tags as candidates.
        Args:
            key (func) : function that given a sentence return an iterable
            valid_values (set): the set of valid values, defaults to None.
        """
        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])
            """shift：单个词的距离（注：不是token）
                原代码是['Northern','Ireland'],若为['Northern Ireland']-->会导致计算出错！
                build_topic_graph() weights.append(1.0 / gap)-->fix
            words: ['Boris Johson','went','to','New York'], [token.text for token in sentence] spacy_doc.sents
            原代码: self.length = len(words) 修改为--> 
                    len([w for word in words for w in word.split()])
            """
            #print()
            #print('   --shift-->',i,shift,sentence.words)

            # container for the sequence (defined as list of offsets)
            seq = []
            seq_offset = [] # for ['Northern Ireland'], 原版本['Northern','Ireland']
            words_num = 0
            # loop through the tokens, key=lambda s: s.pos-->key(self.sentences[i])-->pos[]
            for j, value in enumerate(key(self.sentences[i])):#['Boris Johson','went','to','New York']
                token_text = sentence.words[j]#['Boris Johson','went','to','New York']
                words_num0 = words_num
                words_num += len(token_text.split())
                #print(j,token_text,words_num)                
                # add candidate offset in sequence and continue if not last word
                if value in valid_values:
                    """原有代码 会将name entity与其他可能单词连成一个candidate
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue                    
                    """
                    seq.append(j)
                    seq_offset.append(words_num0)
                    #check if including name entity --> try to remove ne from candidate
                    if j == (len(key(self.sentences[i])) - 1): # the last
                        pass
                    #'a major step in relaxing the public health guidelines Americans have lived with for more than a year'
                    #                                               dobj      nsubj --> 不该成为1个keyword phrase
                    #I see all the   time       people saying
                    #               npadvmod    nsubj --> 不该成为1个keyword phrase
                    elif (sentence.dep[j]=='dobj' or sentence.dep[j]=='npadvmod') and \
                        (sentence.dep[j+1]=='nsubj' or sentence.dep[j+1]=='nsubjpass'):
                        pass
                    #Please join MIT SMR authors Bart de Langhe and Stefano Puntoni.
                    #                     dobj    appos --> 不跟前面的word组成keyword phrase
                    # appositional modifier,同位词 
                    elif sentence.dep[j]=='dobj' and sentence.dep[j+1]=='appos':
                        #print('- - - dobj + appos - - -')
                        #print(token_text,'|',sentence.words[j+1])
                        #print(sentence.words)                        
                        pass
                    elif len(token_text.split())==1:# single word token不检测是不是name entity
                        continue
                    elif token_text in self.ents_list:# this token is name entity
                        pass
                    elif not (sentence.words[j+1] in self.ents_list and 
                                len(sentence.words[j+1].split())>=2):# next token not name entity(>=2 words)
                        continue
                    """    
                    if token_text in self.ents_list:# this token is name entity
                        pass
                    #elif j < (sentence.length - 1):# not name entity, and not the last token
                    elif j < (len(key(self.sentences[i])) - 1):
                        if not (sentence.words[j+1] in self.ents_list):# next token not name entity
                            continue
                    """
                    #"""
                # add sequence as candidate if non empty
                if seq:
                    #print(sentence.words[seq[0]:seq[-1] + 1])                    
                    """
                    tempwords = [w for ww in sentence.words[seq[0]:seq[-1] + 1] for w in ww.split()]
                    #index_word2token = 
                    tempindex = []
                    if len(tempwords)>1:
                        print(sentence.words[seq[0]:seq[-1] + 1])
                        print(tempwords)
                        for ent in self.ents_list:
                            ent_words = ent.split()
                            for i in range(len(tempwords)):
                                if tempwords[i:i+len(ent_words)]==ent_words:
                                    print(i,',',ent_words)
                    """
                    # add the ngram to the candidate container
                    temp_offsets = sentence.meta['char_offsets']
                    #t0 = temp_offsets[0][0]
                    #temp_offsets = [(t[0]-t0,t[1]-t0) for t in temp_offsets]#按句子归一化offset
                    self.add_candidate(words=sentence.words[seq[0]:seq[-1] + 1],
                                       stems=sentence.stems[seq[0]:seq[-1] + 1],
                                       pos=sentence.pos[seq[0]:seq[-1] + 1],
                                       #offset=shift + seq[0],
                                       offset=shift + seq_offset[0],
                                       sentence_id=i,
                                       candidate_char_offset=(temp_offsets[seq[0]][0],temp_offsets[seq[-1]][1])
                                       )

                # flush sequence container
                seq = []
                seq_offset = []               
            
        #print(list(self.candidates))
        #exit()

    def grammar_selection(self, grammar=None):
        """Select candidates using nltk RegexpParser with a grammar defining
        noun phrases (NP).

        Args:
            grammar (str): grammar defining POS patterns of NPs.
        """

        # initialize default grammar if none provided
        if grammar is None:
            grammar = r"""
                NBAR:
                    {<NOUN|PROPN|ADJ>*<NOUN|PROPN>} 
                    
                NP:
                    {<NBAR>}
                    {<NBAR><ADP><NBAR>}
            """

        # initialize chunker
        chunker = RegexpParser(grammar)

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # convert sentence as list of (offset, pos) tuples
            tuples = [(str(j), sentence.pos[j]) for j in range(sentence.length)]

            # parse sentence
            tree = chunker.parse(tuples)

            # find candidates
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    leaves = subtree.leaves()

                    # get the first and last offset of the current candidate
                    first = int(leaves[0][0])
                    last = int(leaves[-1][0])

                    # add the NP to the candidate container
                    self.add_candidate(words=sentence.words[first:last + 1],
                                       stems=sentence.stems[first:last + 1],
                                       pos=sentence.pos[first:last + 1],
                                       offset=shift + first,
                                       sentence_id=i)

    @staticmethod
    def _is_alphanum(word, valid_punctuation_marks='-'):
        """Check if a word is valid, i.e. it contains only alpha-numeric
        characters and valid punctuation marks.

        Args:
            word (string): a word.
            valid_punctuation_marks (str): punctuation marks that are valid
                    for a candidate, defaults to '-'.
        """
        #for name entity token with spaces, such as 'boris johnson'
        word = word.replace(' ', '')

        for punct in valid_punctuation_marks.split():
            word = word.replace(punct, '')
        return word.isalnum()

    #从topicrank.py移过来
    def candidate_selection(self, pos=None, stoplist=None):
        """Selects longest sequences of nouns and adjectives as keyphrase
        candidates.

        Args:
            pos (set): the set of valid POS tags, defaults to ('NOUN',
                'PROPN', 'ADJ').
            stoplist (list): the stoplist for filtering candidates, defaults to
                the nltk stoplist. Words that are punctuation marks from
                string.punctuation are not allowed.

        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = self.stoplist

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                          ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] +
                                          stoplist)

    def candidate_filtering(self,
                            stoplist=None,
                            minimum_length=sgzconstant.MINIMUM_CHAR_LENGTH,#3,
                            minimum_word_size=sgzconstant.MINIMUM_WORD_CHAR_LENGTH,#2,
                            valid_punctuation_marks=sgzconstant.VALID_PUNCTUATION_MARKS,#以空格作为间隔
                            maximum_word_number=sgzconstant.MAXIMUM_WORD_NUMBER,#5,
                            only_alphanum=True,
                            pos_blacklist=None,
                            discard_stoplist=True,
                            ):
        """Filter the candidates containing strings from the stoplist. Only
        keep the candidates containing alpha-numeric characters (if the
        non_latin_filter is set to True) and those length exceeds a given
        number of characters.
            
        Args:
            stoplist (list): list of strings, defaults to None.
            minimum_length (int): minimum number of characters for a
                candidate, defaults to 3.
            minimum_word_size (int): minimum number of characters for a
                token to be considered as a valid word, defaults to 2.
            valid_punctuation_marks (str): punctuation marks that are valid
                for a candidate, defaults to '-'.
            maximum_word_number (int): maximum length in words of the
                candidate, defaults to 5.
            only_alphanum (bool): filter candidates containing non (latin)
                alpha-numeric characters, defaults to True.
            pos_blacklist (list): list of unwanted Part-Of-Speeches in
                candidates, defaults to [].
        """
        #print('-----------------')
        #print(len(list(self.candidates)))
        #print(list(self.candidates))
        #exit()

        if stoplist is None:
            stoplist = []

        if pos_blacklist is None:
            pos_blacklist = []

        # loop through the candidates
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # get the words from the first occurring surface form
            words = [u.lower() for u in v.surface_forms[0]]
            newwords = [word for token in words for word in token.split()] #lower case
            
            """
            print('-------------------')
            print(k,words,newwords,v.pos_patterns[0])
            
            print(v.lexical_form,len(v.lexical_form),
                sum([len(token.split()) for token in v.lexical_form]),'words个数 <=',maximum_word_number)            
            print(words,len(''.join(newwords)),'所有character个数 >=',minimum_length)
            print(min([len(u) for u in newwords]),'单个word的最小character数 >=',minimum_word_size)
            """
            # discard if single word and ADJ (get POS from the first occurring)
            if len(v.pos_patterns[0])==1 and v.pos_patterns[0][0]=='ADJ':
                #print('       ADJ:',words)
                del self.candidates[k]

            # discard if words are in the stoplist
            #if set(words).intersection(stoplist):
            elif discard_stoplist and set(newwords).intersection(stoplist):
                #print(v.lexical_form,words,newwords)
                if not any(newwords==ent.lower().split() for ent in self.ents_list): #check if this is NOT a name entity
                    #print('   *** Not a name entity !!! Deleting ***')
                    del self.candidates[k]

            # discard if tags are in the pos_blacklist
            elif set(v.pos_patterns[0]).intersection(pos_blacklist):
                del self.candidates[k]

            # discard if containing tokens composed of only punctuation
            #elif any([set(u).issubset(set(punctuation)) for u in words]):
            elif any([set(u).issubset(set(punctuation)) for u in newwords]):
                del self.candidates[k]

            # discard candidates composed of 1-2 characters
            #elif len(''.join(words)) < minimum_length:
            elif len(''.join(newwords)) < minimum_length:
                del self.candidates[k]

            # discard candidates containing small words (1-character)
            #elif min([len(u) for u in words]) < minimum_word_size:
            elif min([len(u) for u in newwords]) < minimum_word_size:
                del self.candidates[k]

            # discard candidates composed of more than 5 words
            #elif len(v.lexical_form) > maximum_word_number:
            elif sum([len(token.split()) for token in v.lexical_form]) > maximum_word_number:
                del self.candidates[k]

            # discard if not containing only alpha-numeric characters
            if only_alphanum and k in self.candidates:
                if not all([self._is_alphanum(w, valid_punctuation_marks)
                            for w in newwords]):#words]):
                    del self.candidates[k]

