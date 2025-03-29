# ======== runMinHashExample =======
# This example code demonstrates comparing documents using the MinHash
# approach.
#
# First, each document is represented by the set of shingles it contains. The
# documents can then be compared using the Jaccard similarity of their
# shingle sets. This is computationally expensive, however, for large numbers
# of documents.
#
# For comparison, we will use the MinHash algorithm to calculate short
# signature vectors to represent the documents. These MinHash signatures can
# then be compared quickly by counting the number of components in which the
# signatures agree. We'll compare all possible pairs of documents, and find
# the pairs with high similarity.
#


import sys
import random
import time
from hashlib import sha1  # given hash function

# This is the number of components in the resulting MinHash signatures.
# Correspondingly, it is also the number of random hash functions that
# we will need in order to calculate the MinHash.
numHashes = 10

# Read data
numDocs = 1000
dataFile = "../data/articles_1000.train"
truthFile = "../data/articles_1000.truth"

# =============================================================================
#                  Parse The Ground Truth Tables
# =============================================================================
# Build a dictionary mapping the document IDs to their plagiaries, and vice-
# versa.
plagiaries = {}

# Open the truth file.
f = open(truthFile, "rU")

# For each line of the files...
for line in f:

    # Strip the newline character, if present.
    if line[-1] == "\n":
        line = line[0:-1]  # remove the '\n'

    docs = line.split(" ")  # 0 & 1: two similar docs

    # Map the two documents to each other.
    plagiaries[docs[0]] = docs[1]
    plagiaries[docs[1]] = docs[0]

# =============================================================================
#               Convert Documents To Sets of Shingles
# =============================================================================

print("Shingling articles...")

# Create a dictionary of the articles, mapping the article identifier (e.g.,
# "t8470") to the list of shingle IDs that appear in the document.
docsAsShingleSets = {}  # key: docID, value: set of shingle IDs in the doc

# Open the data file.
f = open(dataFile, "r")

docNames = []

t0 = time.time()

totalShingles = 0

for i in range(0, numDocs):

    # Read all of the words (they are all on one line) and split them by white
    # space.
    words = f.readline().split(" ") # split to many words, every 3 words is a shingle 

    # Retrieve the article ID, which is the first word on the line.
    docID = words[0]

    # Maintain a list of all document IDs.
    docNames.append(docID)

    del words[0] # delete the docID

    # 'shinglesInDoc' will hold all of the unique shingle IDs present in the
    # current document. If a shingle ID occurs multiple times in the document,
    # it will only appear once in the set (this is a property of Python sets).
    shinglesInDoc = set()

    ######## TODO ########
    # Convert the article into a shingle set
    # Each shingle should contain 3 tokens
    # You should use sha1 (imported from hashlib at the beginning of this file) as the hash function that maps shingles into 32-bit integer.
    # You may use int.from_bytes and hashlib._HASH.digest methods
    for index in range(len(words) - 2):
        shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]  # get one shingle
        shingle = shingle.encode("utf-8")  # encode the shingle
        shingleID = int.from_bytes(sha1(shingle).digest(), byteorder="big") # generate byte hash and convert to int
        shinglesInDoc.add(shingleID)

    ##### end of TODO #####

    # Store the completed list of shingles for this document in the dictionary.
    docsAsShingleSets[docID] = shinglesInDoc

    # Count the number of shingles across all documents.
    totalShingles = totalShingles + (len(words) - 2)   # n-k+1=n-2

# Close the data file.
f.close()

# Report how long shingling took.
print("\nShingling " + str(numDocs) + " docs took %.2f sec." % (time.time() - t0))

print("\nAverage shingles per doc: %.2f" % (totalShingles / numDocs))  # the number of shingles per doc average

# =============================================================================
#            Define Matrices to Store extimated JSim
# =============================================================================

### define a table `estJSim` to store estimated Jsim  ###
### Hint for efficiency:
### http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
### The Triangular-Matrix Method section

estJSim = [None for _ in range(numDocs * (numDocs - 1) // 2)] # the number of pairs of docs Cn2


def getTriangleIndex(i, j) -> int:
    if i < j:
        return int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1
    elif i > j:
        return getTriangleIndex(j, i)
    else:
        raise ValueError("Indices must be different")


# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================

# Time this step.
t0 = time.time()

print("\nGenerating random hash functions...")

# Record the maximum shingle ID that we assigned.
maxShingleID = 2**32 - 1

# We need the next largest prime number above 'maxShingleID'.
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID, i.e. nextPrime
# For convenience, I have randomly picked them and fix them for you.
a_list = [
    2647644122,
    3144724950,
    1813742201,
    3889397089,
    850218610,
    4228854938,
    3422847010,
    1974054350,
    1398857723,
    3861451095,
]
b_list = [
    2834859619,
    3834190079,
    3272971987,
    1421011856,
    1598897977,
    1288507477,
    1224561085,
    3278591730,
    1664131571,
    3749293552,
]
# So the i-th hash function is
# h(x) = (a_list[i] * x + b_list[i]) % nextPrime

print("\nGenerating MinHash signatures for all documents...")

# List of documents represented as signature vectors
signatures = []

# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.

# For each document...
for docID in docNames: # namely the doc ids

    # Get the shingle set for this document.
    shingleIDSet = docsAsShingleSets[docID]

    # The resulting minhash signature for this document. It is supposed to be a list of ints.
    signature = []

    ###### TODO ######
    # complete the signature of this doc
    # i.e. set the list `signature' to a proper value.
    for i in range(numHashes):
        minHashCode = nextPrime + 1  # larger than nextPrime
        for x in shingleIDSet:
            # use hash function as given
            hx = (a_list[i] * x + b_list[i]) % nextPrime 
            # set the lowest hash code
            if hx < minHashCode:
                minHashCode = hx
        signature.append(minHashCode)  # ultimate hx
    ### end of TODO ###
    # Store the MinHash signature for this document.
    signatures.append(signature)

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0

print("\nGenerating MinHash signatures took %.2fsec" % elapsed)

# =============================================================================
#                     Compare All Signatures
# =============================================================================

print("\nComparing all signatures...")

# Creates a N x N matrix initialized to 0.

# Time this step.
t0 = time.time()

# For each of the test documents...
for i in range(0, numDocs):
    # Get the MinHash signature for document i.
    signature1 = signatures[i]

    # For each of the other test documents...
    for j in range(i + 1, numDocs):

        # Get the MinHash signature for document j.
        signature2 = signatures[j]

        ####### TODO #######
        # calculate the estimated Jaccard Similarity
        # Then store the value into estJSim
        estJSim[getTriangleIndex(i, j)] = sum([1 for x, y in zip(signature1, signature2) if x == y]) / numHashes  
        ##### end of TODO #####

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0

print("\nComparing MinHash signatures took %.2fsec" % elapsed)


# =============================================================================
#                   Display Similar Document Pairs
# =============================================================================

# Count the true positives and false positives.
tp = 0
fp = 0

threshold = 0.5
print("\nList of Document Pairs with J(d1,d2) more than", threshold)
print("Values shown are the estimated Jaccard similarity and the actual")
print("Jaccard similarity.\n")
print("                   Est. J   Act. J")

# For each of the document pairs...
f = open("../data/prediction.csv", "w")
f.write("article1,article2,Est. J,Act. J\n")
for i in range(0, numDocs):
    for j in range(i + 1, numDocs):

        estJ = estJSim[getTriangleIndex(i, j)]

        # If the similarity is above the threshold...
        if estJ > threshold:

            ###### TODO ######
            # Calculate the actual Jaccard similarity between two docs (shingle sets) for validation.
            # J = 0.0  # You should set the actual Jaccard similarity here.
            J = len(docsAsShingleSets[docNames[i]].intersection(docsAsShingleSets[docNames[j]])) / len(docsAsShingleSets[docNames[i]].union(docsAsShingleSets[docNames[j]]))
            ### end of TODO ###

            # Print out the match and similarity values with pretty spacing.
            print("  %5s --> %5s   %.2f     %.2f" % (docNames[i], docNames[j], estJ, J))
            f.write("{},{},{},{}\n".format(docNames[i], docNames[j], estJ, J))

            # Check whether this is a true positive or false positive.
            # We don't need to worry about counting the same true positive twice
            # because we implemented the for-loops to only compare each pair once.

            if docNames[i] in plagiaries and plagiaries[docNames[i]] == docNames[j]:
                tp = tp + 1
            else:
                fp = fp + 1


f.close()
# Display true positive and false positive counts.
print()
print("True positives:  " + str(tp) + " / " + str(int(len(plagiaries.keys()) / 2)))
print("False positives: " + str(fp))
