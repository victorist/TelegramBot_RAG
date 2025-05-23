# Запуск приложения на сервере

При запуске приложения в докере поступило предупреждение: `WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.`

На основании предоставленных сообщений можно выделить несколько моментов, которые стоит учесть:


1. **Предупреждение о разработческом сервере Flask:**
   ```
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
   ```
   Flask предупреждает о том, что его встроенный сервер предназначен для разработки и не рекомендуется использовать его в продакшене. Вместо этого рекомендуется использовать более надежные и производительные WSGI серверы, такие как Gunicorn, для запуска Flask приложений в продакшене.

### Рекомендации:

- **Использование WSGI сервера в продакшене:**
  Подготовьте ваше приложение к запуску в продакшене с использованием WSGI сервера, например Gunicorn. Это обеспечит стабильную и производительную работу вашего приложения.

### Пример использования Gunicorn:

1. Установите Gunicorn:
   ```bash
   pip install gunicorn
   ```

2. Создайте файл `app.py` или `wsgi.py` для запуска вашего Flask приложения:
   ```python
   # app.py или wsgi.py

   from your_module import app  # Замените на имя вашего модуля и объект приложения

   if __name__ == "__main__":
       app.run()
   ```

3. Запустите ваше приложение с помощью Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8080 app:app  # Замените `app` на имя вашего файла и объект приложения
   ```

   Это запустит ваше приложение на порту 8080 с 4 рабочими процессами (`-w 4`). Подробнее о параметрах Gunicorn можно узнать из его документации.

Следуя этим рекомендациям, вы сможете улучшить стабильность и надежность вашего приложения, а также подготовиться к будущим изменениям в используемых библиотеках и инструментах.

# использования нового API LangChain
Для использования нового API LangChain, вам необходимо заменить использование устаревшего класса `LLMChain` на `RunnableSequence`. Вот как можно адаптировать ваш код:

1. **Замена использования `LLMChain` на `RunnableSequence`:**

   Вам нужно будет изменить создание объекта `LLMChain` на создание объекта `RunnableSequence`, который объединяет промпт и языковую модель.

   Вот как выглядит обновленный код:

   ```python
   from langchain import RunnableSequence
   from langchain.prompts import ChatPromptTemplate
   from langchain.models import YandexLLM

   # Предполагается, что у вас уже есть конфигурация и инструкции
   instructions = """
   Ты информационный бот авиакомпании "АЗУР эйр" по имени Роберт. 
   Твоя задача - полно и вежливо отвечать на вопросы собеседника.
   """

   # Создаем промпт для документов
   document_prompt = ChatPromptTemplate(input_variables=["page_content"], template="{page_content}")

   # Создаем языковую модель
   llm = YandexLLM(config=cfg, use_lite=False, temperature=0.1, instruction_text=instructions)

   # Создаем RunnableSequence, объединяя промпт и языковую модель
   runnable_sequence = ChatPromptTemplate() | llm

   # Пример использования
   input_document = {"page_content": "Ваш текст документа здесь"}
   output_text = runnable_sequence(input_document)

   print("Результат:", output_text)
   ```

   Здесь `ChatPromptTemplate()` создает промпт для документов, а `llm` создает объект языковой модели. `ChatPromptTemplate() | llm` объединяет их в `RunnableSequence`, который можно вызывать с входными данными (`input_document`), чтобы получить результат (`output_text`).

2. **Обновление вашего приложения:**

   В вашем приложении измените способ создания цепочки, используя новый подход `RunnableSequence`. Обратите внимание, что вам также потребуется проверить и обновить другие части вашего кода, которые могут использовать устаревшие API LangChain.

3. **Поддержка и документация:**

   Проверьте [официальную документацию LangChain](https://langchain.readthedocs.io/en/latest/) для более подробной информации о новом API и советах по переходу.

Эти изменения помогут вам использовать последнюю версию LangChain и избежать устаревших функций, которые могут быть удалены в будущих релизах.