import nltk

# nltk.download('words')
# nltk.download('movie_reviews')

from nltk.corpus import words

import numpy as np
import time

import hashlib

word_list = words.words()

from nltk.corpus import movie_reviews

reviews = []

for fileid in movie_reviews.fileids("pos"):
    reviews.extend(movie_reviews.words(fileid))

# Ground Truth:
flag_b = np.zeros(len(reviews), dtype=bool)

t = time.time()
word_list_set = set(word_list)
for i, neg_word in enumerate(reviews):
    flag_b[i] = neg_word in word_list_set


class BloomFilter:

    def __init__(self, m, k):
        self.m = int(m)
        self.k = int(k)
        self.table = np.zeros((self.m), dtype=bool)
        self.h = ["bloom_filter" + str(i) for i in range(k)]

    def add(self, item):
        for i in range(self.k):
            hasher = hashlib.sha256(self.h[i].encode())
            hasher.update(item.encode())
            hasher = int(hasher.hexdigest(), 16) % self.m
            self.table[hasher] = 1

    def check(self, item):
        # TODO
        # Check if the item is in the bloom filter
        for i in range(self.k):
            hasher = hashlib.sha256(self.h[i].encode())
            hasher.update(item.encode())
            hasher = int(hasher.hexdigest(), 16) % self.m
            if self.table[hasher] == 0:
                # 有一个不为1就说明没存上
                return False
        return True

        # end of TODO


def BM(m=1e7, k=1):
    bf = BloomFilter(m, k)
    bf_results = np.zeros(len(reviews), dtype=bool)
    for word in word_list:
        bf.add(word)
    for i, word in enumerate(reviews):
        bf_results[i] = bf.check(word)
    print(f"FPR for m={m}, k={k}:")
    FP = ((bf_results == True) & (flag_b == False)).sum()
    N = (flag_b == False).sum()
    print(f"    FP {FP} / N {N} = {FP / N }")


if __name__ == "__main__":
    for k in range(1, 10):
        BM(m=1e6, k=k)
