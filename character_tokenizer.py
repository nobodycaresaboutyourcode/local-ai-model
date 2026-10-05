# character_tokenizer.py
# exercise 2 - character-level tokenization
# borrowed heavily from Angelos Perivolaropoulos (https://github.com/angelos-p/llm-from-scratch)
# @authors: nobodycaresdude with help from Claude Opus 5.5

# this module performs character level tokenization
# on a corpus of data based on the ENRON email dataset
# each unique character in the dataset becomes its own token
text = open("enron_mails.csv").read()
chars = sorted(set(text))

# added to display vocabulary size - not used at this point.
# the vocabulary size for the ENRON data is 125 characters
# vocab_size = len(chars)
# print(f"Vocabulary size: {vocab_size}")

# this accounts for all unique characters in the ENRON dataset (characters, punctuation, numbers, whitespace, etc.)
# and results in 125^2 (15625) possible bigrams
# a bigram is a pair of consecutive characters in the text

# convert characters to indices
stoi = {c: i for i, c in enumerate(chars)}
# convert indices back to characters
itos = {i: c for c, i in stoi.items()}

# function to encode a string into a list of indices
def encode(s):
    return [stoi[c] for c in s]

# function to decode a list of indices back into a string
def decode(ids):
    return "".join([itos[i] for i in ids])

# print the indices for a sample string
# print(encode("Hello, world!"))
# print the decoded string for a sample list of indices
# print(decode([68, 96, 103, 103, 106, 40, 28, 114, 106, 109, 103, 95, 29]))