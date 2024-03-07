import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import nltk
import mlflow

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def remove_urls(text):
    # Eliminar URLs del texto
    return ' '.join(word for word in text.split() if not word.startswith('http'))
 
def remove_numbers(text):
    # Eliminar números del texto
    return ''.join(char for char in text if not char.isdigit())

def remove_html_tags(text):
    # Eliminar etiquetas HTML del texto
    return BeautifulSoup(text, 'html.parser').get_text()

def preprocess_text(text):
    # Eliminar URLs, etiquetas HTML, números y convertir a minúsculas
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_numbers(text)
    text = text.lower()
    return text

def tokenize_and_lemmatize(text):
    # Tokenizar texto
    tokens = word_tokenize(text)
    
    # Etiquetado de partes del discurso
    pos_tags = nltk.pos_tag(tokens)
    
    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token, pos_tag in pos_tags:
        pos = pos_tag[0].lower()  # Convertir etiquetas POS de Penn Treebank a etiquetas WordNet
        pos = pos if pos in ['a', 'r', 'n', 'v'] else None  # Mapeo de etiquetas POS a etiquetas WordNet
        lemma = lemmatizer.lemmatize(token, pos=pos) if pos else token
        lemmatized_tokens.append(lemma)
    
    return ' '.join(lemmatized_tokens)


def main():
    """Función principal del script."""

    # Argumentos de entrada y salida
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Ruta al archivo de datos de entrada")
    parser.add_argument("--text_column", type=str, help="Nombre de la columna de texto")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.20)
    parser.add_argument("--train_data", type=str, help="Ruta para guardar los datos de entrenamiento")
    parser.add_argument("--test_data", type=str, help="Ruta para guardar los datos de prueba")
    args = parser.parse_args()

    # Inicio del registro de MLFlow
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("Datos de entrada:", args.data)

    # Lectura del archivo de datos
    credit_df = pd.read_csv(args.data)

    ##### Preprocesamiento de texto #####

    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    credit_df[args.text_column] = credit_df[args.text_column].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Preprocesar texto (eliminar URLs, etiquetas HTML, números y convertir a minúsculas)
    credit_df[args.text_column] = credit_df[args.text_column].apply(preprocess_text)

    # Tokenizar y lematizar texto
    credit_df[args.text_column] = credit_df[args.text_column].apply(tokenize_and_lemmatize)

    # Vectorizar texto
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(credit_df[args.text_column])

    # Reducir dimensionalidad con SVD truncado
    svd = TruncatedSVD(n_components=100)
    X_svd = svd.fit_transform(X_vectorized)

    credit_df[args.text_column] = X_svd

    #####################################

    # Registrar métricas en MLFlow
    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    # Dividir datos en conjuntos de entrenamiento y prueba
    credit_train_df, credit_test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )

    # Guardar datos de entrenamiento y prueba en los directorios especificados
    credit_train_df.to_csv(os.path.join(args.train_data, "reviews_sample_reduce_20_01.csv"), index=False)
    credit_test_df.to_csv(os.path.join(args.test_data, "reviews_sample_reduce_20_01.csv"), index=False)

    # Fin del registro de MLFlow
    mlflow.end_run()

# 
if __name__ == "__main__":
    main()
