# tokenizer.py
# exercise 2 - word-based tokenization
# borrowed heavily from Angelos Perivolaropoulos (https://github.com/angelos-p/llm-from-scratch)
# @authors: nobodycaresdude with help from Claude Opus 5.5

# this example uses GPT2 encoding to tokenize word pairs instead of characters
# tiktoken provides the GPT2 encoding for tokenizing text
import tiktoken
enc = tiktoken.get_encoding("gpt2")

text = open("enron_mails.csv").read()
# encode the text into GPT2 tokens
tokens = enc.encode(text)

# the ENRON mail dataset results in ~460 million tokens
# yeah, those tokens.
# print(f"Number of tokens: {len(tokens)}")

# there's no need to turn our tokens back into text at this time
# this is just to demonstrate decoding the tokens back into text
# text = enc.decode(tokens)
# print(f"Decoded text: {text[:1000]}")  # print the first 100 characters of the decoded text
