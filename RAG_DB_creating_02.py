import os
from tqdm import tqdm
import json

from langchain_community.embeddings.yandex import YandexGPTEmbeddings

import lancedb
import langchain
import langchain.document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB as LangChainLanceDB

chunk_size = 512
chunk_overlap = 128
source_dir = 'docs4db'
cfg = "config.json"

loader = langchain.document_loaders.DirectoryLoader(source_dir, glob="*.txt", show_progress=True, recursive=True)
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
fragments = splitter.create_documents([x.page_content for x in loader.load()])

print('>>> Кол-во фрагментов текстов в документе:', len(fragments))
print("_" * 80)
print('>>> Фрагмент 0:\n', fragments[0])
print('>>> Фрагмент 1:\n', fragments[1])
print("_" * 80)

def filter_fragments(fragments):
    filtered_fragments = []
    for fragment in fragments:
        if len(fragment.page_content) > 0:
            filtered_fragments.append(fragment)
        else:
            print("!!! Обнаружен фрагмент документа нулевой длины!")
    return filtered_fragments

filtered_fragments = filter_fragments(fragments)

print("_" * 80)
print(">>> Reading config...")
with open(cfg) as f:
    config = json.load(f)

folder_id = config['folder_id']
api_key = config['api_key']

embeddings = YandexGPTEmbeddings(api_key=api_key, folder_id=folder_id)
print("_" * 80)
print(">>> YandexGPTEmbeddings initialized")

db_dir = "store"
os.makedirs(db_dir, exist_ok=True)

db_instance = lancedb.connect(db_dir)

# Проверка эмбеддингов
print(">>> Document's fragments embeddings...")
embeddings_list = []
#for fragment in filtered_fragments:
for fragment in tqdm(filtered_fragments):
    embedding = embeddings.embed_query(fragment.page_content)
    if len(embedding) == 0:
        print(f"!!! Ошибка: эмбеддинг для фрагмента нулевой длины! Фрагмент: {fragment.page_content}")
    embeddings_list.append({
        "vector": embedding,
        "text": fragment.page_content
    })
print(">>> Embedding of the documents finished")

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

q = "Я инвалид-колясочник. Что мне нужно сделать, чтобы мне помогли в аэропорту?"
print("_" * 80)
print(">>> Query:", q)
print(">>> Test similarity_search...")
print("_" * 80)

# Используем LangChain для выполнения поиска
try:
    # Создание LangChainLanceDB instance с правильными параметрами
    langchain_db_instance = LangChainLanceDB(embeddings, db_instance, "vector_index")

    # Выполнение similarity_search
    res = langchain_db_instance.similarity_search(q, search_type="similarity", k=20)  # Указываем тип поиска и k
    for x in res:
        print('~' * 50)
        print(x.page_content)
except Exception as e:
    print(f"Ошибка при выполнении similarity_search: {e}")

print("_" * 80)