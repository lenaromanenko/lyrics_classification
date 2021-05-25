## Lyrics Classification with MultinomialNB

The goal of the project is to build a text classification model on song lyrics and predict the artist from a piece of text.

<p align="center">
<img src="https://github.com/lenaromanenko/lyrics_classification/blob/main/readme_file_images/artists_readme.jpeg" width="1000">
</p>

## CLI
Let's copy a piece of text from Frank Sinatra's famous song from a website and check whether the model can predict the singer right.
<p align="center">
<kbd><img src="https://github.com/lenaromanenko/lyrics_classification/blob/main/readme_file_images/cli_1.gif" width="1000"></kbd>
</p>

**Result:** With a probability of 97%, the model predicts that the singer of the chosen song is Frank Sinatra! :clap: 

See more results [here](https://github.com/lenaromanenko/lyrics_classification/blob/main/readme_file_images/cli_2.gif).

## Workflow:
1. Choosing some artists on [MetroLyrics](https://www.metrolyrics.com/).
2. Web Scraping: downloading the URLs of all songs of chosen artists and getting song lyrics using **Requests module**, **RegEx**, and **BeautifulSoup**.
3. Constructing text corpus (a list of strings) and labels.
4. Cleaning the text with the help of **Natural Language Toolkit (NLTK)** or **spaCy**. There are both text cleaning methods for NLP in the [classification_model.py](https://github.com/lenaromanenko/lyrics_classification/blob/main/command-line%20interface/classification_model.py).
5. Converting a text corpus into a numerical matrix using **Bag of Words method (BoW)**.
6. Normalizing the counts with the **Term Frequency and the Inverse Document Frequency (TF-IDF)**.
7. Applying Naive Bayes algorithm for multinomially distributed data **(MultinomialNB)**. Putting TF-IDF and MultinomialNB in a pipeline.
8. Exporting the code from Jupyter to a Python file and сreating a pipeline for building a [**CLI**](https://github.com/lenaromanenko/lyrics_classification/tree/main/command-line%20interface).
9. Creating [**Word Cloud**](https://github.com/lenaromanenko/lyrics_classification/blob/main/wordcloud/text_classification_and_word_cloud.ipynb) with the most frequent words in songs of chosen artists:

<p align="center">
<kbd><img src="https://github.com/lenaromanenko/lyrics_classification/blob/main/wordcloud/wordcloud_songs.png" width="1000"></kbd>
</p>

