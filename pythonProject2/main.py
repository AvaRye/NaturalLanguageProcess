from nltk.corpus import brown
import matplotlib.pyplot as plt
from nltk.corpus import names
import nltk
import random


# 1
def SOUNDEX(name, size=4):
    soundex_digits = '01230120022455012623010202'
    soundex = ''
    fc = ''
    for c in name.lower():
        if c.isalpha():
            if not fc:
                fc = c
            d = soundex_digits[ord(c) - ord('a')]
            if not soundex or d != soundex[-1]:
                soundex += d
    soundex = fc + soundex[1:]
    soundex = soundex.replace('0', '')
    return (soundex + size * '0')[:size]


print("1")
print(SOUNDEX('abcdefghijklmnopqrstuvwxyz'))
print(SOUNDEX('Tim'))
print(SOUNDEX('Trump'))
print(SOUNDEX('Einstein'))

# 2.1
# nltk.download('brown')
# 使用以下categories
id = 3018216005
categories = brown.categories()[id % len(brown.categories())]
brown_tagged_sents = brown.tagged_sents(categories=categories)
brown_sents = brown.sents(categories=categories)
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print("2.1")
print('categories:', categories)
print('size:', size)
print('train size:', len(train_sents))
print('test size:', len(test_sents))
print(unigram_tagger.evaluate(test_sents))

# 2.2
ratios = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
evaluations = []
for ratio in ratios:
    size2 = int(len(brown_tagged_sents) * ratio)
    train_sents2 = brown_tagged_sents[:size2]
    test_sents2 = brown_tagged_sents[size2:]
    unigram_tagger2 = nltk.UnigramTagger(train_sents2)
    evaluations.append(unigram_tagger.evaluate(test_sents2))
print("2.2")
print('evaluations: ', evaluations)
plt.plot(ratios, evaluations)
plt.show()

# 2.3
y = []
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])
unseen_sent = brown_sents[3203]
bigram_tagger.tag(unseen_sent)
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
y.append(t1.evaluate(test_sents))
t2 = nltk.BigramTagger(train_sents, backoff=t1)
y.append(t2.evaluate(test_sents))
t3 = nltk.TrigramTagger(train_sents, backoff=t2)
y.append(t3.evaluate(test_sents))
print("2.3")
print('evaluations N-gram from 1 to 3:', y)
fig = plt.figure()
plt.bar([1, 2, 3], y, width=0.4)
plt.show()


# 3
def gender_features2(name):
    features = {"firstletter": name[0].lower(), "lastletter": name[-1].lower()}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
    features["has(%s)" % letter] = (letter in name.lower())
    return features


def gender_features3(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}


# nltk.download('names')
print("3")
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in
                                                                 names.words('female.txt')])
random.shuffle(names)
train_names = names[150:]
devtest_names = names[50:150]
test_names = names[:50]
test_set = [(gender_features3(n), g) for (n, g) in test_names]
train_set = [(gender_features3(n), g) for (n, g) in train_names]
devtest_set = [(gender_features3(n), g) for (n, g) in devtest_names]
classifier1 = nltk.NaiveBayesClassifier.train(train_set)
classifier2 = nltk.MaxentClassifier.train(train_set, max_iter=10)
classifier3 = nltk.DecisionTreeClassifier.train(train_set)
print("classifier1 accuracy:", format(nltk.classify.accuracy(classifier1, devtest_set), '.3f'))
print("classifier2 accuracy:", format(nltk.classify.accuracy(classifier2, devtest_set), '.3f'))
print("classifier3 accuracy:", format(nltk.classify.accuracy(classifier3, devtest_set), '.3f'))

# errors = []
# for (name, tag) in devtest_names:
#     guess = classifier.classify(gender_features2(name))
#     if guess != tag:
#         errors.append((tag, guess, name))
# for (tag, guess, name) in sorted(errors):  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
#     print('correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name))
