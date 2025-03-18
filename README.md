OCR Sentiment Analysis & Text Summarization

Overview

This project performs Optical Character Recognition (OCR), Sentiment Analysis, and Text Summarization on input text. It extracts text from images, analyzes the sentiment, and generates a summary.

Features

Extract text from images using OCR

Perform Sentiment Analysis (Positive, Negative, Neutral)

Summarize extracted text for quick insights

Technologies Used

Python

Tesseract-OCR (for text extraction)

NLTK / TextBlob (for sentiment analysis)

Hugging Face Transformers / Gensim (for text summarization)

Flask / FastAPI (for API deployment)

OpenCV / PIL (for image processing)

Installation

Prerequisites

Ensure you have Python 3.x installed.

Steps

Clone the repository:

"git clone https://github.com/tusharthakur8267/OCR_Sentiment_Analysis_Text_Summarization.git
cd OCR_Sentiment_Analysis_Text_Summarization"

Install dependencies:

pip install -r requirements.txt

Install Tesseract-OCR (if not already installed):

Windows: Download from Tesseract GitHub

Linux: Install using

sudo apt install tesseract-ocr

Mac:

brew install tesseract

Usage

Run the script:

python main.py

For API usage:

uvicorn app:app --reload

Upload an image or input text, and get results for:

Extracted Text

Sentiment Analysis

Text Summary

Example

Input Image:



Extracted Text:

This is an amazing project!

Sentiment Analysis:

Positive

Summary:

Amazing project!

Contribution

Feel free to fork this repo, create a branch, and submit a pull request. Contributions are welcome!
