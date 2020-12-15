from bs4 import BeautifulSoup
import urllib.request
import re
from nltk import FreqDist
import nltk

# 2.1
print("2.1")
url = 'http://www.tju.edu.cn/english/About_TJU/History.htm'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 '
                  'Safari/537.36'
}
request = urllib.request.Request(headers=headers, url=url)
response = urllib.request.urlopen(request).read()
soup = BeautifulSoup(response, "html.parser", from_encoding="utf-8").get_text()
pattern1 = '[a-zA-Z]+'
words = re.findall(pattern1, soup)
fdist1 = FreqDist(words)
print(list(fdist1.keys())[:30])

# 2.2
print("2.2")
fdist1.plot(10)

# 3.1
text = open('1.txt', 'r', encoding='UTF-8').read()
# nltk.download('punkt')
# tokens = nltk.word_tokenize(text)
# print(tokens[:10])

# 3.2
print("3.2")
# pretty
pattern2 = 'w[a-zA-Z]*'
ws = re.findall(pattern2, text)
fdist2 = FreqDist(ws)
# print(sorted(list(fdist2.keys()))[3:8])

# single line
print(sorted(list(FreqDist(re.findall('w[a-zA-Z]*', text)).keys()))[3:8])

# 4.1 remove html tags
print("4.1")
s1 = r'<!DOCTYPE html> <html> <head> <title>Title</title> </head> ' \
     r'<body> <h1>Heading</h1> <p>Paragraph.</p> </body> </html>'
s12 = re.sub(r'<[^>]+>', "", s1, re.S)
s13 = re.sub(r'\s{2,}', " ", s12)
print(s13)

# 4.2
print("4.2")
s2 = r'I do not konw what 1321*3+8 is.'
s22 = re.sub(r'[a-zA-Z]', "", s2, re.S)
s23 = re.sub(r'\s*', "", s22, re.S)
s24 = re.sub(r'\.', "", s23, re.S)
print(s24 + " = " + str(eval(s24)))

# 4.3
print("4.3")
s3 = 'ip1=1.2.3.4 ip2=34.5.6 ip3=256.4.3.2 ip4=0.0.0.0'
result = re.findall(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                    s3)
print(result)