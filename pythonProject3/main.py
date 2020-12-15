from nltk.corpus import conll2000
import nltk
from nltk import CFG


# 1.1
# nltk.download('conll2000')
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


class TrigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
postags = sorted(set(pos for sent in train_sents for (word, pos) in sent.leaves()))
print("1.1")
print("unigram_chunker:")
print(unigram_chunker.evaluate(test_sents))

# 1.2
print("1.2")
bigram_chunker = BigramChunker(train_sents)
print("bigram_chunker:")
print(bigram_chunker.evaluate(test_sents))
trigram_chunker = TrigramChunker(train_sents)
print("trigram_chunker:")
print(trigram_chunker.evaluate(test_sents))

# 2.1
sent1 = 'the dog saw a man in the park'.split()
grammar1 = CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> "saw" | "ate" | "walked"
NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
Det -> "a" | "an" | "the" | "my"
N -> "man" | "dog" | "cat" | "telescope" | "park"
P -> "in" | "on" | "by" | "with"
""")
print("2.1")
print(grammar1)
parser1 = nltk.RecursiveDescentParser(grammar1)
for tree in parser1.parse(sent1):
    print(tree)

# 2.2
print("2.2")
sent2 = 'Tom the cat'.split()
grammar2 = CFG.fromstring('''
S -> NP
NP -> NP Det N
Det -> 'the'
NP -> 'Tom'
N -> 'cat'
''')
# parser2 = nltk.RecursiveDescentParser(grammar2)
parser3 = nltk.BottomUpLeftCornerChartParser(grammar2)
for tree in parser3.parse(sent2):
    print(tree)
