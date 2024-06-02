import os
import docx2txt
import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_samples
import numpy as np
import skfuzzy as fuzz
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil

# загрузка списка стоп-слов
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
# инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # токенизация текста
    tokens = word_tokenize(text.lower())

    # фильтрация токенов и лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()
              and word not in stop_words_en and word not in stop_words_ru]

    # объединение токенов обратно в текст
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def extract_text_from_docx(docx_path):
    try:
        # извлечение текста из документа формата .docx
        text = docx2txt.process(docx_path)
        return text
    except Exception as e:
        messagebox.showerror("Ошибка!", f"Произошла ошибка при извлечении текста из Word: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    try:
        # открытие PDF-файла
        with open(pdf_path, 'rb') as file:
            # создание объекта PDFReader
            pdf_reader = PyPDF2.PdfReader(file)
            # инициализация переменной для хранения текста
            text = ""
            # чтение каждой страницы PDF-файла
            for page in pdf_reader.pages:
                # извлечение текста с каждой страницы и добавление его к общему тексту
                text += page.extract_text()
            return text
    except Exception as e:
        messagebox.showerror("Ошибка!", f"Произошла ошибка при извлечении текста из PDF: {e}")
        return None


def load_documents(directory):
    documents = []
    document_names = []
    for filename in os.listdir(directory):
        # обработка документа формата .txt
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(preprocess_text(file.read()))
                document_names.append(filename)
        # обработка документа формата .docx
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(os.path.join(directory, filename))
            if text:
                documents.append(preprocess_text(text))
                document_names.append(filename)
        # обработка документа формата .pdf
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(directory, filename))
            if text:
                documents.append(preprocess_text(text))
                document_names.append(filename)
        # нетекстовые документы игнорируются
        else:
            continue

    return documents, document_names


# кластеризация с использованием нечеткой нейронной сети
def cluster_documents(documents):
    if not documents:
        messagebox.showerror("Ошибка!", "Нет документов для кластеризации.")
        return None, None

    # векторизация текста
    # создание объекта TfidfVectorizer для преобразования текстовых документов в матрицу TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1)
    # преобразование текстовых документов в TF-IDF матрицу
    tfidf_matrix = vectorizer.fit_transform(documents)

    # создание объекта TruncatedSVD для уменьшения размерности TF-IDF матрицы
    svd = TruncatedSVD(n_components=100)
    # применение метода fit_transform к TF-IDF матрице для уменьшения размерности
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # создание массива данных
    data = np.array(tfidf_reduced)
    # критерий останова
    min_delta_fpc = 0.001
    prev_fpc = -np.inf

    # поиск оптимального количества кластеров
    # начальное значение параметра нечеткости
    m = 1.1
    k = 1
    while True:
        # обучение нечеткой нейронной сети
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, c=k, m=m, error=0.005, maxiter=1000, init=None
        )

        # проверка критерия останова
        if abs(fpc - prev_fpc) < min_delta_fpc:
            break

        # обновление параметра нечеткости
        prev_fpc = fpc
        m += 0.1
        k+=1

    # преобразование результатов в метки кластеров
    labels = np.argmax(u, axis=0)

    # преобразование меток в непрерывные индексы начиная с 0
    unique_labels = np.unique(labels)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    continuous_labels = np.array([label_map[label] for label in labels])

    # вычисление силуэта для каждого документа
    if len(set(continuous_labels)) < 2:
        # отдельная обработка случая с одним кластером
        messagebox.showwarning("Предупреждение!", "Выделен всего один кластер.")
        silhouette_values = 0
    else:
        silhouette_values = silhouette_samples(tfidf_matrix, continuous_labels)

    return continuous_labels, silhouette_values


def browse_button():
    global documents_directory
    filename = filedialog.askdirectory()
    documents_directory.set(filename)


def visualize_clusters():
    # получение директории
    directory = documents_directory.get()
    if not directory:
        messagebox.showerror("Ошибка!", "Выберите папку с текстовыми документами.")
        return

    # загрузка файлов
    documents, document_names = load_documents(directory)
    if documents is not None:
        # кластерный анализ
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

            # информация о кластерах и силуэтах в текстовое поле
            text_output.insert(tk.END, "Документы, разбитые на кластеры:\n")
            silhouette_dict = {i: [] for i in set(labels)}
            for doc_name, label, silhouette in zip(document_names, labels, silhouette_values):
                silhouette_dict[label].append((doc_name, silhouette))
            # вывод кластеров по порядку, название документа и его значение силуэта
            for cluster in sorted(silhouette_dict.keys()):
                text_output.insert(tk.END, f"\nКластер {cluster}:\n")
                for doc_name, silhouette in silhouette_dict[cluster]:
                    text_output.insert(tk.END, f"  - {doc_name}: {silhouette:.4f}\n")

            # вычисление общего среднего значения силуэта
            overall_silhouette = sum(silhouette_values) / len(silhouette_values)
            text_output.insert(tk.END, f"\nСреднее значение силуэта: {overall_silhouette:.4f}\n")

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


def create_cluster_folders():
    # получение директории
    directory = documents_directory.get()
    if not directory:
        messagebox.showerror("Ошибка!", "Выберите папку с документами.")
        return

    # определение родительской директории
    parent_directory = os.path.dirname(directory)
    if not parent_directory:
        messagebox.showerror("Ошибка!", "Не удалось определить родительскую директорию.")
        return

    # загрузка файлов
    documents, document_names = load_documents(directory)
    if documents is not None:
        # кластерный анализ
        labels, _ = cluster_documents(documents)
        if labels is not None:
            cluster_dirs = {}
            for label in set(labels):
                # создание папок для кластеров
                cluster_dir = os.path.join(parent_directory, f"Cluster_{label}")
                os.makedirs(cluster_dir, exist_ok=True)
                cluster_dirs[label] = cluster_dir

            for doc_name, label in zip(document_names, labels):
                # копирование файлов в соответствующие папки кластеров
                src_path = os.path.join(directory, doc_name)
                dest_path = os.path.join(cluster_dirs[label], doc_name)
                shutil.copy2(src_path, dest_path)

            messagebox.showinfo("Выполнено", "Папки кластеров успешно созданы и файлы скопированы.")


def delete_cluster_folders():
    # получение директории
    directory = documents_directory.get()
    if not directory:
        messagebox.showerror("Ошибка!", "Выберите папку с документами.")
        return

    # определение родительской директории
    parent_directory = os.path.abspath(os.path.join(directory, os.pardir))
    if not parent_directory:
        messagebox.showerror("Ошибка!", "Не удалось определить родительскую директорию.")
        return

    # удаление директорий кластеров
    documents, document_names = load_documents(directory)
    if documents is not None:
        labels, _ = cluster_documents(documents)
        if labels is not None:
            for label in set(labels):
                cluster_dir = os.path.join(parent_directory, f"Cluster_{label}")
                if os.path.exists(cluster_dir):
                    shutil.rmtree(cluster_dir)

            messagebox.showinfo("Выполнено", "Папки кластеров успешно удалены.")

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
button_frame.pack(fill=tk.BOTH)

cluster_button = tk.Button(button_frame, text="Кластеризация", command=visualize_clusters, bg="grey75", fg="black", font=("Open Sans", 10, "bold"), cursor="hand2")
cluster_button.pack(side=tk.LEFT, padx=(200, 100), pady=15)

create_folders_button = tk.Button(button_frame, text="Создать папки", command=create_cluster_folders, bg="grey75", font=("Open Sans", 10, "bold"), cursor="hand2")
create_folders_button.pack(side=tk.LEFT, padx=(100, 0), pady=15)

clear_button = tk.Button(button_frame, text="Очистить", command=clear_data, bg="grey75", fg="black", font=("Open Sans", 10, "bold"), cursor="hand2")
clear_button.pack(side=tk.RIGHT, padx=(0, 100), pady=15)

delete_folders_button = tk.Button(button_frame, text="Удалить папки", command=delete_cluster_folders, bg="grey75", font=("Open Sans", 10, "bold"), cursor="hand2")
delete_folders_button.pack(side=tk.RIGHT, padx=(100, 200), pady=15)

# рамка для текстового поля и графика
text_frame = tk.LabelFrame(root, text="Кластеры",font=("Open Sans", 10))
text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

graph_frame = tk.LabelFrame(root, text="График", font=("Open Sans", 10))
graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# текстовое поле для вывода информации о кластерах
text_output = tk.Text(text_frame, height=10, width=20)
text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# запуск окна
root.mainloop()
