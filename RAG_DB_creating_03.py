import os
from tqdm import tqdm
import json

from langchain_community.embeddings.yandex import YandexGPTEmbeddings

import lancedb
import langchain
import langchain.document_loaders
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB

# Задаём размер `chunk_size` - нужно выбирать исходя из нескольких показателей:
# - Допустимая длина контекста для эмбеддинг-модели. 
#   Yandex GPT Embeddings на июнь 2024 допускает 2048 токенов, 
#   в то время как многие открытые модели HuggingFace имеют длину контекста 512-1024 токена
# - Допустимый размер окна контекста большой языковой модели. 
#   Если мы хотим использовать в запросе top 3 результатов поиска, 
#   то 3 * chunk_size+prompt_size+response_size должно не превышать длины контекста модели.

chunk_size = 1000    #1024   # 2048
chunk_overlap= 100  # 
source_dir = 'docs4db'
cfg = "config.json"

loader = langchain.document_loaders.DirectoryLoader(source_dir, glob="*.txt", show_progress=True, recursive=True)
splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
fragments = splitter.create_documents([x.page_content for x in loader.load()])

print('>>> Кол-во фрагментов текстов в документе:', len(fragments))
print("_" * 80)

def filter_fragments(fragments):
    filtered_fragments = []
    for fragment in fragments:
        if len(fragment.page_content) > 0:  # or other validation criteria
            filtered_fragments.append(fragment)
        else:
            print("!!! Обнаружен фрагмент документа нулевой длины!")
    return filtered_fragments

filtered_fragments = filter_fragments(fragments)

# ВЫЧИСЛЕНИЕ ЭМБЕДИНГОВ
print("_" * 80)
print(">>> Reading config...")
with open(cfg) as f:
    config = json.load(f)
folder_id = config['folder_id']
api_key = config['api_key']

# устанавливаем модель эмбеддинга
embeddings = YandexGPTEmbeddings(api_key=api_key, folder_id=folder_id)
print("_" * 80)
print(">>> YandexGPTEmbeddings initialized")

# путь для хранения БД
db_dir = "store"
os.makedirs(db_dir, exist_ok=True)

# Первоначально создаётся БД
db_instance = lancedb.connect(db_dir)

# Проверка эмбеддингов
print(">>> Vectorization of fragments of documents...")
embeddings_list = []
for fragment in tqdm(filtered_fragments):
    embedding = embeddings.embed_query(fragment.page_content)
    if len(embedding) == 0:
        print(f"!!! Ошибка: эмбеддинг для фрагмента нулевой длины! Фрагмент: {fragment.page_content}")
    embeddings_list.append({
        "vector": embedding,
        "text": fragment.page_content
    })
print(">>> Embedding of the docduments finished")

# Вставка эмбеддингов в LanceDB
try:
    db_instance.create_table(
        "vector_index",
        data=embeddings_list,
        mode="overwrite",
    )
    print(">>> LanceDB table created...")
except Exception as e:
    print(f"Ошибка при создании таблицы в LanceDB: {e}")

# Проверка работы вопрос-ответ
q = "Я инвалид-колясочник. Что мне нужно сделать, чтобы мне помогли в аэропорту?"
print("_" * 80)
print(">>> Query:", q)

print("_" * 80)
print(">>> Test as_retriever...")

# Используем LangChain для выполнения поиска
try:
    # Создание LangChainLanceDB instance с правильными параметрами
    langchain_db_instance = LanceDB(embeddings.embed_query, db_instance, "vector_index_01")

    # Выполнение similarity_search
    res = langchain_db_instance.similarity_search(q, search_type="similarity", k=20)  # Указываем тип поиска и k
    for x in res:
        print('~' * 50)
        print(x['text'])
except Exception as e:
    print(f"Ошибка при выполнении similarity_search: {e}")

print("*" * 100)