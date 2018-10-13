#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict
import numpy as np

tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int) #counts of all bigrams in input


def preprocess_line(line):
    """Removes all characters that are not in characters in the English alphabet, digits, or periods, replaces all
    digits with 0, and converts all characters to lowercase
    Returns the processed line
    """
    line = re.sub(r'\d', '0', line)
    line = re.sub(r'\n', '', line) # removes newline character at the end of each line
    line = re.sub(r'[^A-Za-z.\s\d]', '', line)
    line = line.lower()
    line = '##' + line + '#' # Adds beginning and end of line markers
    return line

def get_all_trigrams():
    """Generates all possible trigrams from the list of allowable characters, assigns them a default probability of
    0, and returns them as a dictionary"""
    trigrams = {}
    letters = [' ','#','.','0','a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for l1 in letters:
        for l2 in letters:
            for l3 in letters:
                trigrams[l1+l2+l3]=0
    return trigrams

def calculate_perplexity(model, data):
    """Calculates the perplexity of a string given a model
       model: a probability distribution
       data: the string to compute perplexity for
       Returns the computed perplexity
    """
# only calculates perplexity on val data using add_alpha_prob from training data
    H_add_alpha = 0
    for i in range(len(data) - 2):
        trigram = data[i:i+3]
        H_add_alpha = H_add_alpha - log(model[trigram], 2)
    H_avg = H_add_alpha / (len(data)-2)
    perplexity = np.power(2, H_avg)
    return perplexity

def optimize_alpha(filename, tri_counts, bi_counts):
    """Chooses best alpha value by calculating perplexity of different models on validation data
       Parameters: the filename of the validation data, the dictionary of trigram raw counts, and the dictionary of bigram raw counts
       Returns the best value of alpha
    """
    with open(filename) as f:
        val_string = f.read()
        val_string = preprocess_line(val_string)

    alpha_perplexity = {}
    for i in range(1, 20):
        alpha = i * 0.05
        model = calculate_add_alpha_prob(tri_counts, bi_counts, alpha)
        perplexity = calculate_perplexity(model, val_string)
        alpha_perplexity[perplexity] = alpha
    return alpha_perplexity[min(alpha_perplexity.keys())]


def calculate_add_alpha_prob(tri_counts, bi_counts, alpha):
    """Calculates the add-alpha probability distribution
       tri_counts: a dictionary of trigrams and their raw counts
       bi_counts: a dictionary of bigrams and their raw counts
       alpha: a value between 0 and 1
       Returns the alpha-smoothed probabilities in a dictionary
    """
    add_alpha_probs = get_all_trigrams()
    for key in add_alpha_probs:
        history = key[0:2]
        history_count = bi_counts[history]
        if key in tri_counts.keys():
            add_alpha_probs[key] = (tri_counts[key] + alpha)  / (history_count + alpha*30)      # add alpha smoothing
        else:
            add_alpha_probs[key] = alpha / (history_count + alpha*30)
    return add_alpha_probs

def generate_from_LM(distribution):
    """Generates a random sequence of 300 characters based on a language model, for question 4
       distribution: a dictionary representing a probability distribution
       returns the random sequence
    """
    random_sequence = "#"
    for i in range(299):
        bigram = random_sequence[-2:]
        if bigram[-1] == '#':
            random_sequence = random_sequence + "#" # appends a # to the end of a line so that the string generation
            # starts over on a new line
            bigram = random_sequence[-2:]
        random_sequence = random_sequence + append_char(bigram, distribution)
    return random_sequence

def normalize_probs(probs):
    """Normalizes the probabilities so that they sum exactly to 1
       probs: a list of probabilities
       Returns a list of normalized probabilities
    """
    normalized_probs = []
    for prob in probs:
        normalized_probs.append(prob / sum(probs))
    return normalized_probs

def append_char(bigram, distribution):
    """Chooses the character with the highest probability based on the previous two characters
       bigram: the previous two characters
       distribution: a dictionary representing a probability distribution
       Returns the selected character
    """
    possible_chars = []
    probs = []
    for key in distribution.keys():
        if key[0:2] == bigram:
            possible_chars.append(key[-1])
            probs.append(distribution[key])
    normalized_probs = normalize_probs(probs)
    random_list = np.random.choice(possible_chars, size=None, replace=True, p=normalized_probs)
    random_char = random_list[0]
    return random_char

def get_br_en_distribution():
    """Reads in the model-br.en file and stores the trigrams and probabilities in a dictionary
       Returns the dictionary
    """
    br_en_distribution = {}
    with open('model-br.en') as f:
        for line in f:
            line = re.sub(r'\n', '', line)
            trigram = line[0:3]
            prob = float(line[4:])
            br_en_distribution[trigram] = prob
    return br_en_distribution

def get_ng_probabilities(model):
    """Prints all trigrams with history 'ng' and their probabilities for question 3"""
    for key in model:
        if key[0:2] == "ng":
            print("{0}: {1}".format(key, model[key]))

def run_test(model, test_string):
    """Calculates and prints the perplexity of the test document
       model: the probability distribution
       test_string: the string to calculate the perplexity for
    """
    print("Perplexity: {}".format(calculate_perplexity(model, test_string)))

#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open(infile) as f:
    tri_counts.clear()
    bi_counts.clear()
    for line in f:
        line = preprocess_line(line)
        #print(line)
        for j in range(len(line)-(2)):
            trigram = line[j:j+3]
            bigram = line[j:j+2]
            tri_counts[trigram] += 1
            bi_counts[bigram] += 1

#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in sorted(tri_counts.keys()):
    print(trigram, ": ", tri_counts[trigram])
print("Trigram counts in ", infile, ", sorted numerically:")
for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    print(tri_count[0], ": ", str(tri_count[1]))

# The validation data consists of 10% each of the training data for the three languages. This data was not used for
# estimating probabilities
best_alpha = optimize_alpha("val_en.txt", tri_counts, bi_counts) # To compute the best alpha and model for German or
# Spanish, change the name of the text file here and change the parameters in the configuration
best_model = calculate_add_alpha_prob(tri_counts, bi_counts, best_alpha)
get_ng_probabilities(best_model)

# Writes the model probabilities to a file
with open("model.txt", 'a') as outfile:
    for key in best_model:
        outfile.write("{}: {}\n".format(key, best_model[key]))

# expected best_alpha for English = 0.15

print(generate_from_LM(best_model)) # generates from our language model for English
print(generate_from_LM(get_br_en_distribution())) # generates from the model in model-br.en


# run this on test file for question 5
with open("test") as f:
    test_string = f.read()
    test_string = preprocess_line(test_string)
    run_test(best_model, test_string)

# For question 5:
# Perplexity of test file with English model: 9.379
# Perplexity of test file with German model: 27.662
# Perplexity of test file with Spanish model: 26.555