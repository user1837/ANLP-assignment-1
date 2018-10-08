#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict

tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int) #counts of all bigrams in input

#this function currently does nothing.
def preprocess_line(line):
    line = re.sub(r'\d', '0', line)
    line = re.sub(r'\n', '', line) # removes newline character at the end of each line
    line = re.sub(r'[^A-Za-z.\s\d]', '', line)
    line = line.lower()
    return line

print(preprocess_line('¿Sería apropiado que usted, Señora Presidenta, escribiese una carta'))

def calculate_mle_prob(tri_counts, bi_counts):
    mle_probs = {}
    for key in tri_counts.keys():
        history = key[0:2]
        history_count = bi_counts[history]
        mle_probs[key] = tri_counts[key] / history_count
    return mle_probs


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
    for line in f:
        line = preprocess_line(line)
        for j in range(len(line)-(2)):
            if j == 0:
                trigram = '#'+line[j:j+2]
                bigram = '#' + line[j:j+1]
                tri_counts[trigram] += 1
                bi_counts[bigram] += 1

            trigram = line[j:j+3]
            bigram = line[j:j+2]
            tri_counts[trigram] += 1
            bi_counts[bigram] += 1

            if j == len(line)-3:
                trigram = line[j+1:j+3] + '#'
                tri_counts[trigram] += 1
                bigram = line[j+1:j+3]
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


print(calculate_mle_prob(tri_counts, bi_counts))