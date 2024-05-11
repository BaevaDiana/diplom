import os
import docx2txt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from fcmeans import FCM
from sklearn.metrics import silhouette_samples
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    if not documents:
        print("Нет документов для кластеризации.")
        return None, None
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

    # Получение меток кластеров
    labels = fcm.predict(tfidf_reduced)

    # Вычисляем силуэт для каждого документа
    silhouette_values = silhouette_samples(tfidf_matrix, labels)

    return labels, silhouette_values


def browse_button():
    global documents_directory
    filename = filedialog.askdirectory()
    documents_directory.set(filename)


def visualize_clusters():
    directory = documents_directory.get()
    documents, document_names = load_documents(directory)
    labels, silhouette_values = cluster_documents(documents)

    # Очищаем текстовое поле
    text_output.delete(1.0, tk.END)

    # Выводим информацию о кластерах в текстовое поле
    text_output.insert(tk.END, "Документы, разбитые на кластеры:\n")
    for cluster in set(labels):
        cluster_docs = [doc_name for doc_name, label in zip(document_names, labels) if label == cluster]
        text_output.insert(tk.END, f"Кластер {cluster}:\n")
        for doc_name in cluster_docs:
            text_output.insert(tk.END, f"  - {doc_name}\n")

    # Визуализация кластеров
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(range(len(labels)), labels, c=labels, cmap='viridis')
    ax.set_title('Clusters')
    ax.set_xlabel('Documents')
    ax.set_ylabel('Cluster')
    fig.colorbar(scatter, ax=ax, label='Cluster')

    # Вставляем график в Tkinter
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


# Создаем графический интерфейс
root = tk.Tk()
root.title("Кластеризация текстовых документов")


documents_directory = tk.StringVar()

label = tk.Label(root, text="Выберите папку с документами:")
label.pack()

entry = tk.Entry(root, textvariable=documents_directory)
entry.pack()

browse_button = tk.Button(root, text="Выбрать", command=browse_button)
browse_button.pack()

cluster_button = tk.Button(root, text="Кластеризация", command=visualize_clusters)
cluster_button.pack()

# Создаем рамки для текстового поля и графика
text_frame = tk.LabelFrame(root, text="Кластеры")
text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

graph_frame = tk.LabelFrame(root, text="Графики")
graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Создаем текстовое поле для вывода информации о кластерах
text_output = tk.Text(text_frame, height=10, width=50)
text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

root.mainloop()
