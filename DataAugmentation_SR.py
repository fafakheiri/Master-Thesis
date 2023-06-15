# codes were obtained from "https://maelfabien.github.io/machinelearning/NLP_8/"

import nltk
import pandas as pd
nltk.download('wordnet')

from nltk.corpus import wordnet

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)

import random
def synonym_replacement(text, n):

    words = text.split()

    stop_words = [ ".", ",", "!", "?", ";" ]
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

df = pd.read_csv("path_to_csv_file")
df2 = df['text'].apply(lambda x: synonym_replacement(x, n=1))