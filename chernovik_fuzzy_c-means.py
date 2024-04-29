import os
import docx2txt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
import numpy as np

# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Загрузка списка стоп-слов и инициализация лемматизатора
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация текста
    tokens = word_tokenize(text.lower())

    # Удаление пунктуации и стоп-слов, а также лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if
              word.isalnum() and word not in stop_words_en and word not in stop_words_ru]

    # Объединение токенов обратно в текст
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def extract_text_from_docx(docx_path):
    try:
        # Извлечение текста из документа формата .docx
        text = docx2txt.process(docx_path)
        return text
    except Exception as e:
        print(f"Ошибка извлечения текста из {docx_path}: {e}")
        return None


def preprocess_documents(folder_path):
    # , output_file):
    preprocessed_texts = []
    document_text_mapping = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Чтение и предобработка текстового файла
                    text = file.read()
                    preprocessed_text = preprocess_text(text)
                    preprocessed_texts.append(preprocessed_text)
                    document_text_mapping[filename] = preprocessed_text
            elif filename.endswith('.docx'):
                # Извлечение текста из документа формата .docx и его предобработка
                text = extract_text_from_docx(file_path)
                if text:
                    preprocessed_text = preprocess_text(text)
                    preprocessed_texts.append(preprocessed_text)
                    document_text_mapping[filename] = preprocessed_text

    # # Сохранение предобработанных текстов в отдельном файле
    # with open(output_file, 'w', encoding='utf-8') as output_file:
    #     for text in preprocessed_texts:
    #         output_file.write(text + '\n')

    # return preprocessed_texts

    return preprocessed_texts, document_text_mapping


folder_path = "./NIR"
# output_file = "preprocessed_texts.txt"
# preprocess_documents(folder_path, output_file)
# Предобработка документов
preprocessed_texts, document_text_mapping = preprocess_documents(folder_path)

# Создание объекта TF-IDF векторизатора
tfidf_vectorizer = TfidfVectorizer()

# Преобразование предобработанных текстов в TF-IDF признаки
tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Вывод размерности матрицы TF-IDF признаков
print("Размерность матрицы TF-IDF признаков:", tfidf_features.shape)


# Функция для выполнения алгоритма Fuzzy C-means
def fuzzy_cmeans(tfidf_features, num_clusters, m=2, max_iter=100, tol=0.0001):
    n_samples, n_features = tfidf_features.shape

    # Инициализация центров кластеров
    centers = np.random.rand(num_clusters, n_features)

    # Итерационный процесс
    for _ in range(max_iter):
        # Расчет матрицы принадлежности
        distances = np.linalg.norm(tfidf_features[:, np.newaxis] - centers, axis=2)
        membership = 1 / distances ** (2 / (m - 1))
        membership = (membership.T / np.sum(membership, axis=1)).T

        # Обновление центров кластеров
        new_centers = np.dot(membership.T, tfidf_features) / np.sum(membership, axis=0)[:, np.newaxis]

        # Проверка на сходимость
        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    # Определение принадлежности кластерам
    labels = np.argmax(membership, axis=1)

    return labels


# Пример использования
num_clusters = 5  # Количество кластеров
labels = fuzzy_cmeans(tfidf_features.toarray(), num_clusters)

# Вывод меток кластеров для каждого документа
for i, label in enumerate(labels):
    print(f"Документ {i+1} отнесен к кластеру {label}")