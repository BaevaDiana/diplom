import os
import docx2txt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def preprocess_documents(folder_path, output_file):
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

    # Сохранение предобработанных текстов в отдельном файле
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for text in preprocessed_texts:
            output_file.write(text + '\n')

    return preprocessed_texts


# Пример использования
folder_path = "./NIR"
output_file = "preprocessed_texts.txt"
preprocess_documents(folder_path, output_file)
