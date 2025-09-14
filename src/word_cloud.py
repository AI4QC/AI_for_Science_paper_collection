from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.
venues = ['icml', 'neurips','iclr']
lists = []
for i in venues:
  lists.extend([os.path.join(f'../{i}',j) for j in os.listdir(f'../{i}')])
text_science = []
text_ml = []
for i in lists:
  print(i)
  data = pd.read_csv(i,dtype=str)
  text_science.extend(list(data['Application']))
  text_ml.extend(list(data['MLTech']))

text_science = [x for x in text_science if str(x) != 'nan']
text_ml = [x for x in text_ml if str(x) != 'nan']

text_science=" ".join(text_science)
text_ml=" ".join(text_ml)

alice_mask = np.array(Image.open(path.join(d, "heart.png")))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
               stopwords=stopwords, contour_width=3, contour_color='steelblue')

wc.generate(text_science)
wc.to_file(path.join(d, "science.png"))
wc.generate(text_ml)
wc.to_file(path.join(d, "ml.png"))
