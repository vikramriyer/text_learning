import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import HTMLParser
import string
import itertools
import time

df = pd.read_csv('Reviews.csv')
df.drop('Id', inplace=True, axis=1)
df.drop('ProductId', inplace=True, axis=1)
df.drop('HelpfulnessNumerator', inplace=True, axis=1)
df.drop('HelpfulnessDenominator', inplace=True, axis=1)
df.drop('ProfileName', inplace=True, axis=1)
df.drop('Time', inplace=True, axis=1)

print df.columns

df['Text'].fillna('', inplace=True)
df['Summary'].fillna('', inplace=True)
df['UserId'].fillna('', inplace=True)
df['Score'].fillna(df['Score'].mean(), inplace=True)

html_parser = HTMLParser.HTMLParser()
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
vectorizer = TfidfVectorizer()

def decode_characters_utf8(row):
  return row.decode("utf8").encode('ascii','ignore')

def split_joint_words_and_convert_to_lower(row):
  row = " ".join(re.findall('[A-Z][^A-Z]*', row))
  return row.lower()

def remove_punctuations_at_last_or_first(row):
  return ' '.join([word.strip(string.punctuation) for word in row.split(" ")])

def remove_html_tags(row):
  return html_parser.unescape(row)

def remove_urls(row):
	return re.sub(r"http\S+", "", row)

def standardize_words(row):
  return ''.join(''.join(s)[:2] for _, s in itertools.groupby(row))

def remove_stop_words(row):
  return ' '.join([w for w in word_tokenize(row) if not w in stop_words])

def remove_spl_char(row):
  return re.sub('[^a-zA-Z0-9\n\.]', ' ', row)

def stem_words(row):
    return stemmer.stem(row)

def pre_process_data(field):

  pre_processed_data = []

  for row in df[field]:
    row = decode_characters_utf8(row)
    row = split_joint_words_and_convert_to_lower(row)
    row = remove_punctuations_at_last_or_first(row)
    row = remove_html_tags(row)
    row = remove_urls(row)
    row = remove_spl_char(row)
    row = remove_stop_words(row)
    row = stem_words(row)

    pre_processed_data.append(row)

  return pre_processed_data

def vectorize(vectorizer):
    pass

def main():
  
  # Pre processing
  t = time.time()
  pre_processed_text = pre_process_data('Text')
  print "Time on Text: {}".format(t - time.time())
  t = time.time()
  pre_processed_summary = pre_process_data('Summary')
  print "Time on Summary: {}".format(t - time.time())

  # Vectorization
  #resultant_vector = vectorizer.fit_transform()
  #print len(resultant_vector)
  #print vectorizer.get_feature_names()[52020:52040]

if __name__ == "__main__":
  main()