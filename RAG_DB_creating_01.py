# coding: utf-8

# Создание векторной базы данных с документами для 
# вопрос-ответного бота по технологии Retrieval-Augmented Generation на LangChain
# [LangChain](https://python.langchain.com) - это мощная библиотека для работы с большими языковыми моделями и для построения конвейеров обработки текстовых данных. В одной библиотеке присутствуют все элементы, которые помогут нам создать вопрос-ответного бота на наборе текстовых данных: вычисление эмбеддингов, запуск больших языковых моделей для генерации текста, поиск по ключу в векторных базах данных и т.д.

# Для начала, необходимо установить `langchain` и сопутствующие библиотеки.
# - langchain 
# - sentence_transformers 
# - lancedb
# - unstructured
# - langchain-community==0.2.1 (эта версия совместима с yandex-chain 0.0.8)
# - yandex-chain


# ЗАГРУЗКА ДОКУМЕнТОВ И РАЗБИЕНИЕ ИХ НА ЧАСТИ
# Для загрузки документов используется `DirectoryLoader`, который загружает файлы из указанной диреткории. 
# Взможна загрузка различных типов файлов,
# документацию по загрузчикам смотри [здесь](https://langchain-fanyi.readthedocs.io/en/latest/modules/indexes/document_loaders/examples/directory_loader.html). 
# Можно использовать загрузчик *файлов* `UnstructuredFileLoader`, который, по-умолчанию, использует метод `DirectoryLoader` ), 
# В параметрах можно задать точноcть разбиения документов. 
# Загрузчик неструктурированных документов позволяет пользователям передавать strategy параметр, 
# который позволяет unstructured узнать, как разбить документ на разделы. 
# В настоящее время поддерживаются следующие стратегии: "hi_res" (по умолчанию) и "fast". 
# Стратегии разбиения на разделы в высоком разрешении более точны, но требуют больше времени для обработки. 
# Быстрые стратегии позволяют быстрее разбивать документ на разделы, но при этом снижают точность. 
# Не для всех типов документов существуют отдельные стратегии разбиения в высоком разрешении и быстром режиме. 
# Для этих типов документов strategy параметр kwarg игнорируется. 
# В некоторых случаях стратегия высокого разрешения будет заменена на fast, если отсутствует
#  зависимость (например, модель разделения документа).
# 
# **Unstructured** создает разные “элементы” для разных фрагментов текста. 
# По умолчанию фрагменты объединяются их вместе, но можно легко сохранить это разделение, указав mode="elements".
# 
# В настоящее время **Unstructured** поддерживает загрузку текстовых файлов, Powerpoint, HTML, PDF-файлов, изображений и многого другого.
# 
# Для работы **retrival augmented generation** треьуется по запросу найти наиболее релевантные фрагменты исходного текста, 
# на основе которых будет формироваться ответ. Для этого нужно разбить текст на такие фрагменты, 
# по которым мы сможем вычислять эмбеддинг, и которые будут с запасом помещаться во входное окно используемой большой языковой модели.
# 
# Для этого можно использовать механизмы фреймворка **LangChain** - например, `RecursiveCharacterTextSplitter`. 
# Он разбивает текст на перекрывающиеся фрагменты по набору типовых разделителей - абзацы, переводы строк, разделители слов.
# # пример про сплиттер - https://python.langchain.com/docs/integrations/vectorstores/vald#basic-example


import os
from tqdm import tqdm
import json

from langchain_community.embeddings.yandex import YandexGPTEmbeddings

import lancedb
import langchain
import langchain.document_loaders
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
# Попробуем использовать LangChain для выполнения поиска
from langchain.vectorstores import LanceDB as LangChainLanceDB

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

loader = langchain.document_loaders.DirectoryLoader(source_dir,glob="*.txt",show_progress=True,recursive=True)
splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
fragments = splitter.create_documents([ x.page_content for x in loader.load()])


print('>>> Кол-во фрагментов текстов в документе:',len(fragments))
print("_"*80)


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
# Для вычисления эмбеддингов можно взять какую-нибудь модель из репозитория `HuggingFace`, 
# с поддержкой русского языка. `LangChain` содержит свои встроенные средства работы с эмбеддингами, 
# и поддерживает модели из `HuggingFace`: 
# ЭТО ПРИМЕР с сайта:
# https://python.langchain.com/docs/integrations/vectorstores/vald#basic-example


print("_"*80)
print(">>> Reading config...")
with open(cfg) as f:
    config = json.load(f)
folder_id = config['folder_id']
api_key = config['api_key']

# устанавливаем модель эмбеддинга
embeddings = YandexGPTEmbeddings(api_key=api_key, folder_id=folder_id) 
print("_"*80)
print(">>> YandexGPTEmbeddings initialized")

# путь для хранения БД
db_dir = "store"
os.makedirs(db_dir,exist_ok=True)

# Первоначально создаётся БД
# ТРебуется import lancedb
db_instance = lancedb.connect(db_dir)

# No need to create a table here; use the connection directly
# Proceed with indexing the documents
# Далее проиндексируем все выделенные ранее фрагменты нашей документации.
"""
table = db_instance.create_table(
    "vector_index",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)
print(">>> LanceDB table created...")
"""

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
        "vector_index_157",
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

"""
print(">>> Test similarity_search...")
print("_" * 80)

try:
    res = db_instance.similarity_search(q)
    for x in res:
        print('~' * 50)
        print(x.page_content)
except Exception as e:
    print(f"Ошибка при выполнении similarity_search: {e}")
"""

print("_" * 80)
print(">>> Test as_retriever...")

# Используем LangChain для выполнения поиска
try:
    # Создание LangChainLanceDB instance с правильными параметрами
    # langchain_db_instance = LangChainLanceDB(embeddings, db_instance, "vector_index_190")
    langchain_db_instance = LangChainLanceDB(embeddings, filtered_fragments, "vector_index_190")


    # Выполнение similarity_search
    res = langchain_db_instance.similarity_search(q, search_type="similarity", k=20)  # Указываем тип поиска и k
    #res = db_instance.similarity_search(q, search_type="similarity", k=20)  # Указываем тип поиска и k

    for x in res:
        print('~' * 50)
        print(x.page_content)
except Exception as e:
    print(f"Ошибка при выполнении similarity_search: {e}")

print("*"*100)

# **************************************************
