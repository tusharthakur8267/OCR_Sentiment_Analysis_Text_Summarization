# Important imports
from app import app
from flask import request, render_template, url_for
import os
import cv2
import numpy as np
from PIL import Image
import random
import string
import pytesseract
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

from heapq import nlargest


# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Route to home page


@app.route("/", methods=["GET", "POST"])
def index():

    # Execute if request is get
    if request.method == "GET":
        full_filename = 'images/white_bg.jpg'
        return render_template("index.html", full_filename=full_filename)

    # Execute if reuqest is post
    if request.method == "POST":
        image_upload = request.files['image_upload']
        imagename = image_upload.filename
        image = Image.open(image_upload)

        # Converting image to array
        image_arr = np.array(image.convert('RGB'))
        # Converting image to grayscale
        gray_img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        # Converting image back to rbg
        image = Image.fromarray(gray_img_arr)

        # Printing lowercase
        letters = string.ascii_lowercase
        # Generating unique image name for dynamic image display
        name = ''.join(random.choice(letters) for i in range(10)) + '.png'
        full_filename = 'uploads/' + name

        # Extracting text from image
        custom_config = r'-l eng --oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)

        # Remove symbol if any
        characters_to_remove = "!()@—*“>+-/,'|£#%$&^_~"
        new_string = text
        for character in characters_to_remove:
            new_string = new_string.replace(character, "")

        # Converting string into list to dislay extracted text in seperate line
        new_string = new_string.split("\n")

        # Saving image to display in html
        img = Image.fromarray(image_arr, 'RGB')
        img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))

        

        # summarization : 1
#         stopwords = list(STOP_WORDS)

#         nlp = spacy.load('en_core_web_sm')

#         doc = nlp(text2)
#         tokens = [token.text for token in doc]
#         # print(tokens)
#         punctuation = ""
#         punctuation = punctuation + '\n'
#         # punctuation

#         word_frequencies = {}
#         for word in doc:
#             if word.text.lower() not in stopwords:
#                 if word.text.lower() not in punctuation:
#                     if word.text not in word_frequencies.keys():
#                         word_frequencies[word.text] = 1
#                     else:
#                         word_frequencies[word.text] += 1

#         # print(word_frequencies)

#         max_frequency = max(word_frequencies.values())
#         # max_frequency

#         for word in word_frequencies.keys():
#                 word_frequencies[word] = word_frequencies[word]/max_frequency

#         # print(word_frequencies)

#         sentence_tokens = [sent for sent in doc.sents]
#         # print(sentence_tokens)

#         sentence_scores = {}

#         for sent in sentence_tokens:
#             for word in sent:
#                 if word.text.lower() in word_frequencies.keys():
#                     if sent not in sentence_scores.keys():
#                         sentence_scores[sent] = word_frequencies[word.text.lower()]
#                     else:
#                         sentence_scores[sent] += word_frequencies[word.text.lower()]

#         # sentence_scores
#         select_length = int(len(sentence_tokens)*0.3)
#         # select_length
#         summary = nlargest(select_length, sentence_scores,key=sentence_scores.get)
#         # summary
#         final_summary = [word.text for word in summary]
#         summary = ' '.join(final_summary)
#         t1=" ".join(summary.split())
#         # TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown'])
#         t3 =TreebankWordDetokenizer().detokenize(final_summary)
#         samm = t1

#         # print(type(t1))
#         print(t1)
#         print(t3)
#         print(summary)
#         print(text2)
#         # print("hello my name is tushar")
#         # print(text2)
# #
        # summarization : 2
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text)

        # Creating a frequency table to keep the
        # score of each word

        freqTable = dict()
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

            # Creating a dictionary to keep the score
        # of each sentence
        sentences = sent_tokenize(text)
        sentenceValue = dict()

        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]

        # Average value of a sentence from the original text

        average = int(sumValues / len(sentenceValue))

        # Storing sentences into our summary.
        summary = ''
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence
        print(summary)


        # Returning template, filename, extracted text
        return render_template('index.html', full_filename=full_filename, text=new_string, text2=summary)


# Main function
if __name__ == '__main__':
    app.run(debug=True)
