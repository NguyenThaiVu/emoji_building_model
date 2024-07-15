import os
import numpy as np
import pandas as pd 
import random
from googletrans import Translator
import nltk
from nltk.corpus import wordnet

def convert_unicode_2_emoji(unicode_str):
    """
    This function take unicode string and return the emoji
    """
    emoji = chr(int(unicode_str[2:], 16))
    return emoji

def convert_emoji_2_unicode_str(emoji):
    unicode_code_point = f'U+{ord(emoji):X}'
    return unicode_code_point


def get_pos(word):
    """Get the part of speech of a word."""
    pos_tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(pos_tag, wordnet.NOUN)


# def get_synonyms_word(word):
#     """
#     This function take word and return list of synonyms
#     """

#     synonyms = set()
#     word_pos = get_pos(word)
    
#     for syn in wordnet.synsets(word):
#         if syn.pos() == word_pos:
#             for lemma in syn.lemmas():
#                 synonyms.add(lemma.name())
    
#     return list(synonyms)


# def sentence_replacement(sentence):
#     """
#     This function replace each word in a sentence by its synonyms
#     e.g.: "i love you" -> "i like you"
#     """

#     list_new_word = []
#     list_words = sentence.split()
#     for word in list_words:
#         list_synonyms = get_synonyms_word(word)
#         if len(list_synonyms) > 0:
#             new_word = random.choice(list_synonyms)
#         else:
#             new_word = word  # in case can not replace this word
#         list_new_word.append(new_word)
#     return " ".join(list_new_word)


# def random_swap(sentence):
#     """
#     This function randomly swap 2 words in the sentence
#     e.g.: "like you" -> "you like"
#     """
#     list_words = sentence.split()
#     if len(list_words) < 2:
#         return sentence
#     idx1, idx2 = random.sample(range(len(list_words)), 2)
#     list_words[idx1], list_words[idx2] = list_words[idx2], list_words[idx1]
#     return ' '.join(list_words)


# def back_translation(text, src_lang='en', tgt_lang='vi'):
#     """
#     This function translate sentence to another language and then back to original language.
#     """

#     translator = Translator()
#     translated = translator.translate(text, src=src_lang, dest=tgt_lang).text
#     back_translated = translator.translate(translated, src=tgt_lang, dest=src_lang).text
#     return back_translated


# def augment_text(sentence, max_iter=3):

#     iter = 0
#     while 1:
#         iter += 1
#         if iter > max_iter:  break

#         seed = random.choice([1, 2, 3])
#         if seed == 1:
#             new_sentence = sentence_replacement(sentence)
#         elif seed == 2:
#             new_sentence = random_swap(sentence)
#         elif seed  == 3:
#             new_sentence = back_translation(sentence)
#         else:
#             new_sentence = sentence
        
#         if sentence != new_sentence:   break  # Find new text

#     return new_sentence



def load_glove_embeddings(file_path):
    """Load GloVe embeddings from a file."""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def sentence_to_vector(sentence, embeddings):
    """
    This function convert a sentence to vector using GloVe embeddings.
    We convert each word into embedding then calculate average of every word in a sentence.
    """
    vectors = []
    for word in sentence.split():
        vector = embeddings.get(word)
        if vector is not None:
            vectors.append(vector)
    
    if len(vectors) == 0:
        return np.zeros(50)  # Return a zero vector if no words are found
    
    return np.mean(vectors, axis=0)


def get_prediction_emotion(xgb_model, input_name, embedding, label_encoder):
    """
    This function take input as name and return its predicted emotion
    """
    processed_name = sentence_to_vector(input_name, embedding)
    processed_name = np.expand_dims(processed_name, axis=0)

    y_pred_label = xgb_model.predict(processed_name)  # Inference

    # convert from label (187) to emotion unicode -> (U+1F913)
    y_pred_unicode = label_encoder.inverse_transform(y_pred_label)[0]
    y_pred_emotion = convert_unicode_2_emoji(y_pred_unicode)

    return y_pred_emotion