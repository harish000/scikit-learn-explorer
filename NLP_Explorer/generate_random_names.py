# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:40:51 2016

@author: Sriharish
"""

# -*- coding: utf-8 -*-

from nltk import *
import random
names = corpus.names
male_names = names.words('male.txt')
male_name_list = [list(w) for w in male_names]
male_name_flatten = [item for sublist in male_name_list for item in sublist]
male_bigram = list(bigrams(male_name_flatten))
def generate_random_model(cfdist, char, count, num=(random.randint(6,11))):
    generated_names = []
    name = []
    for j in range(20):
        ch = char
        ''.join(name)
        generated_names.append(name)
        char = ch
        name = []
        for i in range(num):
            name.append(ch.lower())
            ch = list(cfd[ch].most_common(random.randint(10,20))[random.randint(1,6)])[0]
    return generated_names
cfd = ConditionalFreqDist(male_bigram)
generated_names = generate_random_model(cfd, 'A', 20)
generated_names = [''.join(list).title() for list in generated_names]
del generated_names[0]
print(generated_names)