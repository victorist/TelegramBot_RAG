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
print("Фрагмент 0:", fragments[0])


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
    #langchain_db_instance = LangChainLanceDB(embeddings, db_instance, "vector_index")


    # Выполнение similarity_search
    #res = langchain_db_instance.similarity_search(q, search_type="similarity", k=20)  # Указываем тип поиска и k
    res = db_instance.similarity_search(q, search_type="similarity", k=20)  # Указываем тип поиска и k

    for x in res:
        print('~' * 50)
        print(x.page_content)
except Exception as e:
    print(f"Ошибка при выполнении similarity_search: {e}")

print("*"*100)

# **************************************************



# Ok ВКЛЮЧАЕМ В КОД #################################
#q="Можно ли провозить лыжи на борту самолёта?"
'''
print("_"*80)
print(">>> Test as_retriever...")
retriever = db_instance.as_retriever(search_kwargs={"k": 20} )
# res = retriever.get_relevant_documents(q)
res = retriever.invoke(q)
'''


concatdocs = '\n'
for x in res:
    print(x.page_content)
    print('________')


#    concatdocs += x.page_content

#print(concatdocs)





# Ok #################################
# ВКЛЮЧАЕМ В КОД импорт библиотек (скорее всего достаточно импортировтаь from langchain.prompts import ChatPromptTemplate )
# КОД ИЗ ДОКУМЕНТАЦИИ yandex-chain
# НЕ СКЛЮЧАеМ С КОД ________________________________________
from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

'''
# Отвечай на вопросы только в контексте следующих документв:
#  Answer the question based only on the following context:
template = """ Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = YandexLLM(config="config_neurocat.json", use_lite=False)

chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)
'''


# In[ ]:


# код мз документации НЕ СРАБОТАЛ
#query = 'Можно ли провозить аккумуляторы в салоне самолёта?'
chain.invoke(q)


# Теперь попросим модель ответить на наш вопрос от лица сотрудника компании:

# In[ ]:


# Ok - включаем в код #################################

instructions = """
Ты информационный бот авиакомпании "АЗУР эйр" по имени Роберт. 
Твоя задача - полно и вежливо отвечать на вопросы собеседника.
"""

llm = YandexLLM(config="config.json", use_lite=False, temperature= 0.2,         #  api_key=api_key, folder_id=folder_id,
                instruction_text = instructions)


# In[ ]:


# Ok  не нужно включать в код ________________________________________
# Это просто проверка работы модели без привязки к контенту документов
q = 'Можно ли провозить аккумуляторы в салоне самолёта?'
print('Вопрос клиента:', q)
print('Ответ АзурБота:\n',llm(q))


# В данном примере мы пока что никак не использовали наши текстовые документы.
# 
# ## Собираем Retrieval-Augmented Generation
# 
# Пришла пора соединить всё вместе и научить бота отвечать на вопросы, подглядывая в наш текстовый корпус. Для этого используем механизм цепочек (*chains*). Основную функциональность реализует `StuffDocumentsChain`, которая делает следующее:
# 
# 1. Берёт коллекцию документов `input_documents`
# 1. Каждый из них пропускает через `document_prompt`, и затем объединяет вместе.
# 1. Данный текст помещается в переменную `document_variable_name`, и передаётся большой языковой модели `llm`
# 
# В нашем случае `document_prompt` не будет модицицировать документ, а основной запрос к LLM будет содержать в себе инструкции для Yandex GPT. 

# In[ ]:


# Промпт для обработки документов ### ВКЛЮЧАЕМ В КОД
# Ответ на вопрос выдаётся в контексте релевантных документов (fragments), сохранённых в res
from langchain import chains

document_prompt = langchain.prompts.PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
    )

# Промпт для языковой модели
document_variable_name = "context"
'''
stuff_prompt_override = """
Пожалуйста, отвечай на вопрос только в контексте правил воздушных перевозок авиакомпании "АЗУР эйр".
Если ответа в правилах нет, извениьсь и порекомендуй позвонить в контактный ценр по телефону +7(495)374-55-14 или отправить запрос на электронную почту `call@azurair.ru`.
Если собеседник не задал вопрос, поприветствуй его и сообщи о своей готворности ответить на его вопросы.
Вот правила воздушных перевозок:
-----
{context}
-----
Вопрос:
{query}"""
'''
stuff_prompt_override = """
Отвечай на вопрос только в контексте правил воздушных перевозок авиакомпании "АЗУР эйр"
Если в правилах нет ответа на вопрос собеседника, рекомендуй позвонить в контактный ценр по телефону +7(495)374-55-14 или отправить запрос на электронную почту `call@azurair.ru
При ответе учитывай нижеследующие выдержки из правил воздушных перевозок авиакомпании "АЗУР эйр":
-----
{context}
-----
Вопрос:
{query}"""

prompt = langchain.prompts.PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
    )

# Создаём цепочку
llm_chain = langchain.chains.LLMChain(llm=llm, prompt=prompt)
chain = langchain.chains.StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    )


from pathlib import Path

# Запуск цепочки
results = chain.run(input_documents=res, query=q)

# Создание пути к файлу
path = Path("answer.md")

# Запись результата в файл
with open(path, "w") as file:
    file.write(results)

# Сообщение о завершении
print(f"Результат записан в файл: {path}")





# Чтобы ещё более улучшить результат, мы можем использовать хитрый механизм перемешивания документов, таким образом, чтобы наиболее значимые документы были ближе к началу запроса. Также мы оформим все операции в одну функцию `answer`:

# In[ ]:


# OK ==== Сортировка по релевантрости работает
# Разобраться, куда код вставить

from langchain.document_transformers import LongContextReorder
#reorderer = LongContextReorder(temperature=0.05, device='cuda', return_full_response=True)
reorderer = LongContextReorder()

# "Температура" модели. Чем выше значение, тем более "творческими" будут результаты. По умолчанию: 1.0
# return_full_response (bool, optional) Если True, функция будет возвращать полный ответ модели, включая logits, probabilities и hidden states. По умолчанию: False

def answer(query,reorder=True,print_results=False): # если reordr=True, то производится сортировка по степени релевантности
  results = retriever.get_relevant_documents(query)
  if print_results:
        for x in results:
            print(f"{x.page_content}\n--------")
  if reorder:
    results = reorderer.transform_documents(results)
  return chain.run(input_documents=results, query=query)


# In[ ]:


result2 = answer(q, reorder=True, print_results=True)

# Создание пути к файлу
path = Path("answer_v2.md")

# Запись результата в файл
with open(path, "w") as file:
    file.write(result2)

# Сообщение о завершении
print(f"Результат записан в файл: {path}")


# In[ ]:


# Пример от Gemini ### НЕ ВКЛЮЧАЕМ В КОД
from langchain.document_transformers import LongContextReorder


model = LongContextReorder()

response = model("This is an example sentence.")

print(response)


# In[ ]:


# Пример от Gemini ### НЕ ВКЛЮЧАЕМ В КОД
from langchain.document_transformers import LongContextReorder

chain = (
    chain.map(LongContextReorder())
    .map(lambda x: x.text)
)

response = chain("This is an example sentence.")

print(response)


# Можно сравнить результаты, выдаваемые Yandex GPT напрямую, с результатами нашей генерации на основе документов:


# ### НЕ ВКЛЮЧАЕМ В КОД

def compare(q):
    print(f"Ответ YaGPT: {llm(q)}")
    print(f"Ответ бота: {answer(q)}")
    
compare("Что ты можешь сказать правилах перевозки лыж в самолёте?")

