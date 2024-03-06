import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def remove_urls(text):
    # Remove URLs from text
    return re.sub(r'http\S+', '', text)

def remove_html_tags(text):
    # Remove HTML tags from text
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_numbers(text):
    # Remove numbers from text
    return re.sub(r'\d+', '', text)

def preprocess_text(text):
    # Remove URLs, HTML tags, numbers, and convert to lowercase
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_numbers(text)
    text = text.lower()
    return text

def tokenize_and_lemmatize(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token, pos_tag in pos_tags:
        pos = pos_tag[0].lower()  # Convert Penn Treebank POS tags to WordNet POS tags
        pos = pos if pos in ['a', 'r', 'n', 'v'] else None  # Map POS tags to WordNet tags
        lemma = lemmatizer.lemmatize(token, pos=pos) if pos else token
        lemmatized_tokens.append(lemma)
    
    return ' '.join(lemmatized_tokens)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--text_column", type=str, help="name of the text column")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.20)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    credit_df = pd.read_csv(args.data, header=1, index_col=0)


    ##### Text Preprocessing #####

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    credit_df[args.text_column] = df[args.text_column].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Preprocess text (remove URLs, HTML tags, numbers, and convert to lowercase)
    credit_df[args.text_column] = df[args.text_column].apply(preprocess_text)

    # Tokenize and lemmatize text
    credit_df[args.text_column] = df[args.text_column].apply(tokenize_and_lemmatize)

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(df[args.text_column])

    # Reduce dimensionality with SVD
    svd = TruncatedSVD(n_components=100)
    X_svd = svd.fit_transform(X_vectorized)

    credit_df[args.text_column] = X_svd

    ############################

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)


    credit_train_df, credit_test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    credit_train_df.to_csv(os.path.join(args.train_data, "reviews_job.csv"), index=False)

    credit_test_df.to_csv(os.path.join(args.test_data, "reviews_job.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
