import os
import numpy as np
import pandas as pd 
import random
from googletrans import Translator
import nltk
from nltk.corpus import wordnet




def get_pos(word):
    """Get the part of speech of a word."""
    pos_tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(pos_tag, wordnet.NOUN)


def get_synonyms_word(word):
    """
    This function take word and return list of synonyms using nltk wordnet
    """

    synonyms = set()
    word_pos = get_pos(word)
    
    for syn in wordnet.synsets(word):
        if syn.pos() == word_pos:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    
    return list(synonyms)


def sentence_synonyms_replacement(sentence):
    """
    This function replace each word in a sentence by its synonyms (using nltk library)
    e.g.: "i love you" -> "i like you"
    """

    list_new_word = []
    list_words = sentence.split()
    for word in list_words:
        list_synonyms = get_synonyms_word(word)
        if len(list_synonyms) > 0:
            new_word = random.choice(list_synonyms)
        else:
            new_word = word  # in case can not replace this word
        list_new_word.append(new_word)
    return " ".join(list_new_word)



# To load model:
# model = api.load("glove-wiki-gigaword-50")
def get_similar_glove_words(word, glove_similar_model, topn=3):
    """
    This function take word and return list of similar word using GloVe
    """
    similar_words = glove_similar_model.most_similar(word, topn=topn)
    if similar_words != None and len(similar_words):
        return [word for word, _ in similar_words]
    else:
        return []


def sentence_similar_replacement(sentence, glove_similar_model):
    """
    This function replace each word in a sentence by its similar
    e.g.: "i love you" -> "i like you"

    *Argment:
    model -- a gensim GloVe model, which can be loaded by: model = gensim.downloader.api.load("glove-wiki-gigaword-50")
    """

    list_new_word = []
    list_words = sentence.split()
    for word in list_words:
        list_similars = get_similar_glove_words(word, glove_similar_model)
        if len(list_similars) > 0:
            new_word = random.choice(list_similars)
        else:
            new_word = word  # in case can not replace current word
        list_new_word.append(new_word)
    return " ".join(list_new_word)


def random_swap(sentence):
    """
    This function randomly swap 2 words in the sentence
    e.g.: "like you" -> "you like"
    """
    list_words = sentence.split()
    if len(list_words) < 2:
        return sentence
    idx1, idx2 = random.sample(range(len(list_words)), 2)
    list_words[idx1], list_words[idx2] = list_words[idx2], list_words[idx1]
    return ' '.join(list_words)


def back_translation(text, src_lang='en', tgt_lang='vi'):
    """
    This function translate sentence to another language and then back to original language.
    """

    translator = Translator()
    translated = translator.translate(text, src=src_lang, dest=tgt_lang).text
    back_translated = translator.translate(translated, src=tgt_lang, dest=src_lang).text
    return back_translated


def augment_text(sentence, glove_similar_model, max_iter=3):

    iter = 0
    while 1:
        iter += 1
        if iter > max_iter:  break
        try:
            seed = random.choice([1, 2, 3, 4])
            if seed == 1:
                new_sentence = sentence_synonyms_replacement(sentence)
            elif seed == 2:
                new_sentence = random_swap(sentence)
            elif seed  == 3:
                new_sentence = back_translation(sentence)
            elif seed == 4:
                new_sentence = sentence_similar_replacement(sentence, glove_similar_model)
            else:
                new_sentence = sentence
            
            if sentence != new_sentence:   break  # Find new text
        except:
            pass

    return new_sentence
