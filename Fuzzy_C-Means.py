import os
import docx2txt
import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from fcmeans import FCM
from sklearn.metrics import silhouette_samples
import tkinter as tk
from tkinter import filedialog, messagebox
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
        messagebox.showerror("Ошибка!", f"Произошла ошибка при извлечении текста из Word: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    try:
        # Открытие PDF-файла
        with open(pdf_path, 'rb') as file:
            # Создание объекта PDFReader
            pdf_reader = PyPDF2.PdfReader(file)
            # Инициализация переменной для хранения текста
            text = ""
            # Чтение каждой страницы PDF-файла
            for page in pdf_reader.pages:
                # Извлечение текста с каждой страницы и добавление его к общему тексту
                text += page.extract_text()
            return text
    except Exception as e:
        messagebox.showerror("Ошибка!", f"Произошла ошибка при извлечении текста из PDF: {e}")
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
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(directory, filename))
            if text:
                documents.append(preprocess_text(text))
                document_names.append(filename)
        else:
            messagebox.showerror("Ошибка!", "В папке должны быть текстовые документы.")
            return None, None
    return documents, document_names


def cluster_documents(documents):
    if not documents:
        messagebox.showerror("Ошибка!", "Нет документов для кластеризации.")
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
    # silhouette_values = silhouette_samples(tfidf_matrix, labels)

    # Если у нас есть только один кластер, обработаем его отдельно
    if len(set(labels)) < 2:
        messagebox.showwarning("Предупреждение!", "Всего один кластер.")
        silhouette_values = 0
    else:
        # Вычисляем силуэт для каждого документа
        silhouette_values = silhouette_samples(tfidf_matrix, labels)

    return labels, silhouette_values


def browse_button():
    global documents_directory
    filename = filedialog.askdirectory()
    documents_directory.set(filename)


def visualize_clusters():
    directory = documents_directory.get()
    if not directory:
        messagebox.showerror("Ошибка!", "Выберите папку с текстовыми документами.")
        return

    documents, document_names = load_documents(directory)
    if documents is not None:
        labels, silhouette_values = cluster_documents(documents)
        if labels is not None and silhouette_values is not None:
            if len(set(labels)) < 2:
                messagebox.showwarning("Предупреждение!", "Всего один кластер. График невозможно построить.")
                text_output.delete(1.0, tk.END)
                text_output.insert(tk.END, "Документы, разбитые на кластеры:\n")
                cluster_docs = [doc_name for doc_name, label in zip(document_names, labels)]
                text_output.insert(tk.END, f"Кластер 0:\n")
                for doc_name in cluster_docs:
                    text_output.insert(tk.END, f"  - {doc_name}\n")
                return

            text_output.delete(1.0, tk.END)

            # информация о кластерах в текстовое поле
            text_output.insert(tk.END, "Документы, разбитые на кластеры:\n")
            for cluster in set(labels):
                cluster_docs = [doc_name for doc_name, label in zip(document_names, labels) if label == cluster]
                text_output.insert(tk.END, f"Кластер {cluster}:\n")
                for doc_name in cluster_docs:
                    text_output.insert(tk.END, f"  - {doc_name}\n")

            # визуализация кластеров
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            scatter = ax.scatter(range(len(labels)), labels, c=labels, cmap='viridis')
            ax.set_title('Clusters')
            ax.set_xlabel('Documents')
            ax.set_ylabel('Cluster')
            fig.colorbar(scatter, ax=ax, label='Cluster')

            # отображение графиков
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# кнопка "Очистить"
def clear_data():
    text_output.delete(1.0, tk.END)
    documents_directory.set("")
    global graph_frame
    graph_frame.destroy()
    graph_frame = tk.LabelFrame(root, text="График", font=("Open Sans", 10))
    graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)


# настройка параметров окна
root = tk.Tk()
root.title("Кластеризация текстовых документов")
root.geometry("1300x700+{}+{}".format((root.winfo_screenwidth() - 1300)// 2, (root.winfo_screenheight() - 700) // 2))
root.configure(bg="azure4")

documents_directory = tk.StringVar()

label = tk.Label(root, text="Выберите папку с документами:", bg="ivory3",font=("Open Sans", 11))
label.pack(side=tk.TOP,pady=5)

entry = tk.Entry(root, textvariable=documents_directory, font=("Open Sans", 10), width=50)
entry.pack(padx=7, pady=5)

browse_button = tk.Button(root, text="Выбрать", command=browse_button, bg="grey75", font=("Open Sans", 10, "bold"), cursor="hand2")
browse_button.pack(padx=7, pady=5)

# фрейм для кнопок
button_frame = tk.Frame(root, bg="azure4")
button_frame.pack(fill=tk.X)

cluster_button = tk.Button(button_frame, text="Кластеризация", command=visualize_clusters, bg="grey75", fg="black", font=("Open Sans", 10, "bold"), cursor="hand2")
cluster_button.pack(side=tk.LEFT, padx=225, pady=15)

clear_button = tk.Button(button_frame, text="Очистить", command=clear_data, bg="grey75", fg="black", font=("Open Sans", 10, "bold"), cursor="hand2")
clear_button.pack(side=tk.RIGHT, padx=225, pady=15)

# рамка для текстового поля и графика
text_frame = tk.LabelFrame(root, text="Кластеры",font=("Open Sans", 10))
text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

graph_frame = tk.LabelFrame(root, text="График", font=("Open Sans", 10),)
graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# текстовое поле для вывода информации о кластерах
text_output = tk.Text(text_frame, height=10, width=20)
text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

root.mainloop()