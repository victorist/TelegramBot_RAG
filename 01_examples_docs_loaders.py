# Для работы загрузчика документов из директории необходимо установить
# следующие пакеты
# - unstructured
# - python-libmagic
# - poppler-utils   - images and PDFs
# - pytesseract
# - langchain-community

#from PIL import Image
#import pytesseract

# информация о загрузчиках:
# https://python.langchain.com/v0.2/docs/integrations/document_loaders/
# https://python.langchain.com/v0.1/docs/integrations/providers/unstructured/

# загрузчик документов
from langchain_community.document_loaders import UnstructuredFileLoader
# загрузчик HTML-страниц
from langchain_community.document_loaders import UnstructuredURLLoader

# ________________________________________________
# Демонстрация загрузки различных типов документов
print("_"*80)
print("\nDownloading and parsing PDF-file...")
loader = UnstructuredFileLoader("txts/Буклет-Автоматизация-АК.pdf", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов PDF-документа:")
print(docs[:5])

'''не поддерживается формат PDF-сканов
print("_"*80)
print("\nDownloading and parsing PDF-scan...")
loader = UnstructuredFileLoader("txts/scan_АрутиновАЮ.pdf", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов PDF-скана:")
print(docs[:5])
'''

print("_"*80)
print("\nDownloading and parsing DOCX-file...")
loader = UnstructuredFileLoader("txts/СД 12.12. Положение о порядке заключения договоров.docx", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов DOCX-документа:")
print(docs[:5])

"""не поддерживается этот формат загрузчиком файла
print("\nDownloading and parsing DOC-file...")
loader = UnstructuredFileLoader("txts/Приказ+о+корп+моб+связи.doc", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов DOC-документа:")
print(docs[:5])
"""
print("_"*80)
print("\nDownloading and parsing XLSX-file...")
loader = UnstructuredFileLoader("txts/Azurair_ПК_v2.xlsx", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов XLSX-документа:")
print(docs[:5])

print("_"*80)
print("\nDownloading and parsing PPTX-file...")
loader = UnstructuredFileLoader("txts/Презентация ИТ ИЮЛЬ 2024г.pptx", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов PPTX-документа:")
print(docs[:5])

''' не поддерживается этот формат загрузчиком файла
print("\nDownloading and parsing ODS-file...")
loader = UnstructuredFileLoader("txts/пример_индексов_рисков.ods", mode="elements")
docs = loader.load()
print("Вывод 5-ти элементов ODS-документа:")
print(docs[:5])
'''

# ________________________________________________
# Демонстрация загрузки web-страницы
urls = [
    "https://azurair.ru/ru/for-passengers/fly-rules",
    
]

print("_"*80)
print("\nDownloading and parsing web-site...")
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print("Вывод 5-ти элементов web-сайта:")

print(data[0])

# _____________________________________________________
# Демонстрация загрузки картинки
from langchain_community.document_loaders.image import UnstructuredImageLoader

print("_"*80)
print("\nDownloading and parsing image...")
loader = UnstructuredImageLoader("txts/СУБП Волга-Днепр.png")

data = loader.load()
print("Вывод картинки:")
print(data[0])

# Image captions
# Need >>> pip install --upgrade --quiet  transformers
from langchain_community.document_loaders import ImageCaptionLoader

print("\nImage captions...")

list_image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Aeroflot_Ilyushin_Il-96-300_RA-96011_Mishin.jpg"
    ]

loader = ImageCaptionLoader(images=list_image_urls)
list_docs = loader.load()
print("\nList of images:",list_docs)

import requests
from PIL import Image

Image.open(requests.get(list_image_urls[0], stream=True).raw).convert("RGB")

# Create the index
from langchain.indexes import VectorstoreIndexCreator
print("\nCreate the index...")
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What's the painting about?"
print("\nВопрос:", query)
print("\nОтвет:")
print(index.query(query))