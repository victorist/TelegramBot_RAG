# Creating DB for RAG bot
# source .anvrag/bin/activate

import langchain
# рекомендация >>> import langchain.document_loaders заменить на from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter # ?нужно ли заменить на CharacterTextSplitter из langchain_text_splitters
#from langchain_text_splitters import CharacterTextSplitter

from lancedb.rerankers import LinearCombinationReranker

from tqdm import tqdm
import datetime

# M2
#import metall   # применение Metall на M2


#>documents = CharacterTextSplitter().split_documents(documents)

model_path = "/Users/victoristratov/devrepos/PrivateLLaMa/llmodel/llama-2-7b.Q4_0.gguf"

embeddings = LlamaCppEmbeddings(model_path = model_path)

chunk_size = 1000    #1024   # 2048
chunk_overlap= 100  # 
source_dir = 'docs4db'
cfg = "config.json"

# проверка фрагментов текста на пустые (длиной =0)
def filter_fragments(fragments):
    filtered_fragments = []
    for fragment in fragments:
        if len(fragment.page_content) > 0:  # or other validation criteria
            filtered_fragments.append(fragment)
        else:
            print("!!! Обнаружен фрагмент документа нулевой длины!")
    return filtered_fragments
# _________________________________________________

print('>>> Загрузка документов и деление на фрагменты...')

loader = DirectoryLoader(source_dir,glob="*.txt",show_progress=True,recursive=True)
#splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#fragments = splitter.create_documents([ x.page_content for x in loader.load()])

documents = loader.load()

fragments = CharacterTextSplitter().split_documents(documents)

print('>>> Кол-во фрагментов текстов в документе:',len(fragments))
print("Фрагмент 0:", fragments[0])
print("_"*80)

# проверка на пустые фрагменты
print("Проверка размера фрагментов...")
filtered_fragments = filter_fragments(fragments)

print('>>> Кол-во фрагментов после проверки их длины:',len(filtered_fragments))
print("Фрагмент 0:", filtered_fragments[0])
print("_"*80)



# инициализация векторного хранилища
print("_"*80,"\n>>> Vector Store initialisation...")
vector_store = LanceDB(
    embedding=embeddings,
    table_name='langchain_test'
    )
print(">>> Vector Store initialisated...")


# embadding
print(">>> Reranking...")
reranker = LinearCombinationReranker(weight=0.3)

current_time = datetime.datetime.now().time()
print(">>> Embedding process started at", current_time)
# Предположим, что `filtered_fragments` содержит объекты с атрибутом `page_content`
# Предположим, что `filtered_fragments` может содержать строки или объекты
class Document:
    def __init__(self, page_content):
        self.page_content = page_content

# Проверим, являются ли элементы `filtered_fragments` строками или объектами
if isinstance(filtered_fragments[0], str):
    text_fragments = [Document(content) for content in filtered_fragments]
else:
    text_fragments = filtered_fragments

docsearch = LanceDB.from_documents(filtered_fragments, embeddings, reranker=reranker)

current_time = datetime.datetime.now().time()
print('>>> Embedding process finished at', current_time)


query = "Я инвалид-колясочник. Что мне нужно сделать, чтобы мне помогли в аэропорту?"
print("_" * 80)
print(">>> Query:", query)

print("_" * 80)
print(">>> Similarity search with relevance scores - ...")
docs = docsearch.similarity_search_with_relevance_scores(query)
print("relevance score - ", docs[0][1])
print("text- ", docs[0][0].page_content[:1000])

print("_" * 80)
print(">>> Similarity search with relevance distance...")
docs = docsearch.similarity_search_with_score(query="Headaches", query_type="hybrid")
print("distance - ", docs[0][1])
print("text- ", docs[0][0].page_content[:1000])

print("_" * 80)
print(">>> Save to dataframe...")
tbl = docsearch.get_table()
print("tbl:", tbl)
pd_df = tbl.to_pandas()