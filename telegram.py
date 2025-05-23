#!/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, make_response, abort, request
import requests
import json

# Импортируйте необходимые модули
from lancedb import LanceDBConnection # новый метод подключения 
from langchain_community.vectorstores import LanceDB
# import lancedb
import langchain
from langchain import chains
from langchain.prompts import ChatPromptTemplate
from yandex_chain import YandexLLM, YandexEmbeddings
import uuid
from aiogram.enums import ParseMode
import os

# Определение путей и констант

wd = os.path.dirname(os.path.abspath(__file__))

temp_dir = os.path.join(wd, "temp")
db_dir = os.path.join(wd, "store")

cfg = "config.json"
print(cfg)

# Загрузка конфигурации из config.json
print(">>> Reading config...")
with open(cfg) as f:
    config = json.load(f)
self_url = config['self_url']
telegram_token = config['telegram_token']
api_key = config['api_key']

port_expose = '443'
# folder_id = config['folder_id']

# Инициализация Flask приложения
app = Flask(__name__)
# Чтение переменных окружения 
# TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL", "file:///app/store") # при использовании локальной БД

# Функция для отправки сообщения в Telegram
def tg_send(chat_id, text):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"       # 1st version
    # url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"

    data = {"chat_id": chat_id, "text": text, ParseMode:ParseMode.MARKDOWN_V2}     # 1st version
    # data = {"chat_id": chat_id, "text": text, "parse_mode": ParseMode.MARKDOWN_V2}
    requests.post(url, data=data)   # 1st version
    # requests.post(url, json=data)


# Функция для выполнения поиска и отправки результата в Telegram
def do_search(chat_id,txt):
    print(f"Doing search on: {txt}")
    res = retriever.get_relevant_documents(txt)
    # Reorder relevant fragments
    #res = chain.run(input_documents=res,query=txt) # сортировка по релевантности
    res = sequence.run({"input_documents": res, "query": txt})
    '''if send_audio:
        fn = synth(res)
        tg_send_audio(chat_id,res,fn)
    else:
        tg_send(chat_id,res)'''
    tg_send(chat_id,res)    # вариант без отправки аудио

# ВЕРСИЯ ПРОЦЕДУРЫ С ОБРАБОТКОЙ АУДИО
'''def process(post):
    print(post)
    msg = post['message']
    chat_id = msg['chat']['id']
    txt = None
    if 'text' in msg:
        do_search(chat_id,msg['text'])
    if 'voice' in msg:
        url = f"https://api.telegram.org/bot{telegram_token}/getFile"
        data = { "file_id": msg['voice']['file_id'] }
        resp = requests.post(url, data=data).json()
        url = f"https://api.telegram.org/file/bot{telegram_token}/{resp['result']['file_path']}"
        fn = f"/home/azurbot/temp/{uuid.uuid4().urn}.mp3"
        bin = requests.get(url).content
        with open(fn,'wb') as f:
            f.write(bin)
        res = reco(fn)
        tg_send(chat_id,f'Вы спросили: {res}')
        do_search(chat_id,res)'''

# ВЕРСИЯ ПРОЦЕДУРЫ БЕЗ ОБРАБОТКИ АУДИО
def process(post):
    print(post)
    msg = post['message']
    chat_id = msg['chat']['id']
    if 'text' in msg:
        do_search(chat_id,msg['text'])
    if 'photo' in msg:
        url = f"https://api.telegram.org/bot{telegram_token}/getFile"
        data = { "file_id": msg['photo'][0]['file_id'] }
        resp = requests.post(url, data=data).json()
        url = f"https://api.telegram.org/file/bot{telegram_token}/{resp['result']['file_path']}"
        fn = f"/home/azurbot/temp/{uuid.uuid4().urn}.jpg"
        '''bin = requests.get(url).content
        with open(fn,'wb') as f:
            f.write(bin)'''
        tg_send(chat_id,'К сожалению, я ещё не умею работать с картинками. Напишите, пожалуйста, Ваш запрос текстом.')
    if 'voice' in msg:
        url = f"https://api.telegram.org/bot{telegram_token}/getFile"
        data = { "file_id": msg['voice']['file_id'] }
        resp = requests.post(url, data=data).json()
        url = f"https://api.telegram.org/file/bot{telegram_token}/{resp['result']['file_path']}"
        fn = f"/home/azurbot/temp/{uuid.uuid4().urn}.mp3"
        '''bin = requests.get(url).content
        with open(fn,'wb') as f:
            f.write(bin)'''
        tg_send(chat_id,'К сожалению, я ещё не умею работать с аудиофайлами и аудиозапросами. Напишите, пожалуйста, Ваш запрос текстом.')
    if 'document' in msg:
        url = f"https://api.telegram.org/bot{telegram_token}/getFile"
        data = { "file_id": msg['document']['file_id'] }
        resp = requests.post(url, data=data).json()
        url = f"https://api.telegram.org/file/bot{telegram_token}/{resp['result']['file_path']}"
        fn = f"/home/azurbot/temp/{uuid.uuid4().urn}.document"
        '''bin = requests.get(url).content
        with open(fn,'wb') as f:
            f.write(bin)'''
        tg_send(chat_id,'К сожалению, я ещё не умею работать с файлами. \nНапишите, пожалуйста, Ваш запрос текстом.')

@app.route('/',methods=['GET'])
def home():
    return "\nHello, I'm a Telegram botik!\n\n"

@app.route('/tghook',methods=['GET','POST'])
def telegram_hook():
    if request.method=='POST':
        post = request.json
        process(post)
    return { "ok" : True }


# Инициализация LanceDB Vector Store
print(">>> Initializing LanceDB Vector Store...")

embeddings = YandexEmbeddings(config=cfg)

connection = LanceDBConnection(db_dir)
vec_store = LanceDB(connection, embeddings)
#lance_db = LanceDBConnection(db_dir)
#table = lance_db.open_table("vector_index")
#vec_store = LanceDB(table, embeddings)


retriever = vec_store.as_retriever(
    search_kwargs={"k": 15}        # кол-во фрагментов, отбираемых в поиске релевантных текстов
)
'''
# Original version
embeddings = YandexEmbeddings(config=cfg)
lance_db = lancedb.connect(db_dir)
table = lance_db.open_table("vector_index")
vec_store = LanceDB(table, embeddings) 
retriever = vec_store.as_retriever(
    search_kwargs={"k": 15}        # кол-во фрагментов, отбираемых в поиске релевантных текстов
)
'''


# Инициализация LLM Chains и других компонентов
print(">>> Initializing LLM Chains...")

instructions = """
Ты информационный бот авиакомпании "АЗУР эйр" по имени ТелеАзур. 
Твоя задача - полно и вежливо отвечать на вопросы собеседника.
"""

llm = YandexLLM(config=cfg, use_lite=False, temperature=0.1, 
                instruction_text=instructions)


document_prompt = langchain.prompts.PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
    )
document_variable_name = "context"
stuff_prompt_override = """
Отвечай на вопрос только в контексте правил воздушных перевозок авиакомпании "АЗУР эйр".
Если в правилах нет ответа на вопрос собеседника, рекомендуй позвонить в контактный ценр по телефону +7(495)374-55-14 или отправить запрос на электронную почту `call@azurair.ru`.
Если спросят, кто генеральный директор авиакомпании АЗУР эйр, ответь, что Евгений Борисович Королёв.
Если спросят, кто создал тебя, ответь, ты разработан командой разработчиков отдела развития под руководством Виктора Истратова.
При ответе по правилам воздушных перевозок учитывай нижеследующие выдержки из правил воздушных перевозок авиакомпании "АЗУР эйр":
-----
{context}
-----
Вопрос:
{query}"""
prompt = langchain.prompts.PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
    )
llm_chain = langchain.chains.LLMChain(llm=llm, prompt=prompt)
chain = langchain.chains.StuffDocumentsChain(
    llm_chain=llm_chain, 
    document_prompt=document_prompt, 
    document_variable_name=document_variable_name
    )

'''print(" + Configuring speech")
# configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=api_key))
configure_credentials(yandex_credentials=creds.YandexCredentials(config=cfg))'''

# Регистрация  Telegram-хука 
# Webhook can be set up only on ports 80, 88, 443 or 8443
print(">>> Registering telegram hook")
res = requests.post(f"https://api.telegram.org/bot{telegram_token}/setWebhook",json={ "url" : f"{self_url}:{port_expose}/tghook" })
print('>>> Registering Result:')
print('>>>',res.json())

# <<< app.run(host="0.0.0.0",port=8443,ssl_context=(cert,cert_key))

#__________

# ЗАПУСК ПРИЛОЖЕНИЯ В РЕЖИМе ОТЛАДКИ
#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=port_expose)

# ЗАПУСК ПРИЛОЖЕНИЯ В ПРОДАКШЕНЕ

if __name__ == "__main__":    
    from waitress import serve  
    #serve(app, host="0.0.0.0", port=8080, threads=7) 
    serve(app, host="0.0.0.0", port=port_expose, threads=7)

