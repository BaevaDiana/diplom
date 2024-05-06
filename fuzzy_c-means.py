import os
import docx2txt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from fcmeans import FCM
from sklearn.metrics import silhouette_samples

# Загрузка списка стоп-слов и инициализация лемматизатора
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация текста
    tokens = word_tokenize(text.lower())

    # Удаление пунктуации и стоп-слов, а также лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()
              and word not in stop_words_en and word not in stop_words_ru]

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


def load_documents(directory):
    documents = []
    document_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(preprocess_text(file.read()))
                document_names.append(filename)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(os.path.join(directory, filename))
            if text:
                documents.append(preprocess_text(text))
                document_names.append(filename)
    return documents, document_names

def cluster_documents(documents):
    # Создание объекта TfidfVectorizer для преобразования текстовых документов в матрицу TF-IDF
    # max_df и min_df устанавливают пороговые значения для отбора слов в TF-IDF матрице
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1)

    # Преобразование текстовых документов в TF-IDF матрицу
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Создание объекта TruncatedSVD для уменьшения размерности TF-IDF матрицы до 100 компонент
    svd = TruncatedSVD(n_components=100)

    # Применение метода fit_transform к TF-IDF матрице для уменьшения размерности
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # Создание объекта FCM (Fuzzy C-Means) для кластеризации уменьшенной TF-IDF матрицы
    fcm = FCM()

    # Применение метода fit к уменьшенной TF-IDF матрице для обучения модели FCM
    fcm.fit(tfidf_reduced)


    # Применение обученной модели FCM к уменьшенной TF-IDF матрице для предсказания кластеров
    return fcm.predict(tfidf_reduced)



if __name__ == "__main__":
    documents_directory = "./TEST"
    documents, document_names = load_documents(documents_directory)
    clusters = cluster_documents(documents)

    # Собираем документы для каждого кластера
    cluster_documents_dict = {}
    for doc_name, cluster in zip(document_names, clusters):
        if cluster not in cluster_documents_dict:
            cluster_documents_dict[cluster] = []
        cluster_documents_dict[cluster].append((doc_name, cluster))

    # Выводим документы в сортировке по кластерам
    print("Документы, разбитые на кластеры:")
    for cluster, docs in sorted(cluster_documents_dict.items()):
        print(f"Кластер {cluster}:")
        for doc_name, _ in docs:
            print(f"  - {doc_name}")

    # Создание TF-IDF матрицы
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1)
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Вычисляем силуэт для каждого документа
    silhouette_values = silhouette_samples(tfidf_matrix, clusters)

    # Создаем словарь для хранения уникальных значений силуэта для каждого кластера
    cluster_silhouette_dict = {}
    for cluster, silhouette_val in zip(clusters, silhouette_values):
        if cluster not in cluster_silhouette_dict:
            cluster_silhouette_dict[cluster] = silhouette_val

    # Выводим номер кластера и его силуэт
    print("Номер кластера и его силуэт:")
    for cluster, silhouette_val in sorted(cluster_silhouette_dict.items()):
        print(f"Кластер {cluster}: Силуэт = {silhouette_val}")