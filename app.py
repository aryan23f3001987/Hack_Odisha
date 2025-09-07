# from flask import Flask, render_template, jsonify, request
# # from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# from src.chat_memory import load_memory, save_memory   # ✅ Import memory helpers
# import os
# import requests  # ✅ For weather API & book API

# app = Flask(__name__)

# # ------------------ Load environment variables ------------------ #
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # ✅ Weather API key

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# # ------------------ Embeddings ------------------ #
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# # ------------------ Pinecone VectorStore ------------------ #
# index_name = "cropdata"
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # ------------------ Chat Model ------------------ #
# chatModel = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# # ------------------ Routes ------------------ #
# @app.route("/")
# def index():
#     # ✅ Clear all session files on refresh
#     sessions_dir = os.path.join(os.path.dirname(__file__), "sessions")
#     if os.path.exists(sessions_dir):
#         for file in os.listdir(sessions_dir):
#             if file.endswith(".json"):
#                 os.remove(os.path.join(sessions_dir, file))
#         print("🗑 Cleared all session JSON files")
#     else:
#         print("⚠ sessions folder not found")

#     return render_template("chat.html")


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     username = request.form.get("username", "guest")  # ✅ Default to guest

#     print(f"User: {msg} | Username: {username}")

#     # ✅ Load previous memory for this user
#     memory = load_memory(username)

#     # ✅ Build memory context string
#     memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in memory])

#     # ✅ Create a prompt that includes memory history
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt + "\n\nConversation history:\n" + memory_context),
#             ("human", "{input}"),
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     # ✅ Get response from RAG chain
#     response = rag_chain.invoke({"input": msg})
#     answer = response["answer"]

#     print("Response:", answer)

#     # ✅ Save new memory
#     memory.append({"role": "user", "content": msg})
#     memory.append({"role": "assistant", "content": answer})
#     save_memory(username, memory)

#     # ✅ Return JSON response
#     return jsonify({"answer": answer})


# # ------------------ Weather Routes ------------------ #
# @app.route("/weather")
# def weather_page():
#     return render_template("weather.html")


# @app.route("/get_weather", methods=["POST"])
# def get_weather():
#     city = request.form.get("city")
#     if not city:
#         return jsonify({"error": "Please provide a city name"}), 400

#     # Call OpenWeatherMap API
#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
#     try:
#         res = requests.get(url)
#         data = res.json()
#         if data.get("cod") != 200:
#             return jsonify({"error": data.get("message", "City not found")}), 404

#         weather_info = {
#             "city": data["name"],
#             "temperature": data["main"]["temp"],
#             "humidity": data["main"]["humidity"],
#             "description": data["weather"][0]["description"],
#             "wind_speed": data["wind"]["speed"]
#         }
#         return jsonify(weather_info)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ------------------ Book Routes (Open Library) ------------------ #
# @app.route("/get_books", methods=["POST"])
# def get_books():
#     query = request.form.get("query", "crop farming")  # default search
#     url = f"https://openlibrary.org/search.json?q={query}&limit=12"

#     try:
#         res = requests.get(url, timeout=5)
#         data = res.json()
#         books = []
#         for doc in data.get("docs", []):
#             books.append({
#                 "title": doc.get("title"),
#                 "author": ", ".join(doc.get("author_name", [])),
#                 "first_publish_year": doc.get("first_publish_year"),
#                 # Embed URL to read book online
#                 "read_url": f"https://openlibrary.org{doc.get('key')}?embed=true"
#             })
#         return jsonify(books)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ------------------ Optional Resources Page ------------------ #
# @app.route("/resources")
# def resources_page():
#     return render_template("resources.html")


# # ------------------ Run Server ------------------ #
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port, debug=False)

# from flask import Flask, render_template, jsonify, request
# # from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# from src.chat_memory import load_memory, save_memory   # ✅ Import memory helpers
# import os
# import requests  # ✅ For weather API & other APIs

# app = Flask(__name__)

# # ------------------ Load environment variables ------------------ #
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # ✅ Weather API key

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# # ------------------ Embeddings ------------------ #
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",   # or "text-embedding-3-large" if needed
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# # ------------------ Pinecone VectorStore ------------------ #
# index_name = "cropdata"
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # ------------------ Chat Model ------------------ #
# chatModel = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# ------------------ Actual App ------------------ #
from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.chat_memory import load_memory, save_memory
import os
import requests

app = Flask(__name__)

# ------------------ Load environment variables ------------------ #
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# ------------------ Embeddings ------------------ #
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------ Pinecone VectorStore ------------------ #
index_name = "cropdata"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------------ Chat Model ------------------ #
chatModel = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------ Routes ------------------ #
@app.route("/")
def index():
    sessions_dir = os.path.join(os.path.dirname(__file__), "sessions")
    if os.path.exists(sessions_dir):
        for file in os.listdir(sessions_dir):
            if file.endswith(".json"):
                os.remove(os.path.join(sessions_dir, file))
        print("🗑 Cleared all session JSON files")
    else:
        print("⚠ sessions folder not found")
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    username = request.form.get("username", "guest")
    print(f"User: {msg} | Username: {username}")

    memory = load_memory(username)
    memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in memory])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nConversation history:\n" + memory_context),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]

    print("Response:", answer)

    memory.append({"role": "user", "content": msg})
    memory.append({"role": "assistant", "content": answer})
    save_memory(username, memory)

    return jsonify({"answer": answer})


# ------------------ Weather Routes ------------------ #
@app.route("/weather")
def weather_page():
    return render_template("weather.html")


@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.form.get("city")
    if not city:
        return jsonify({"error": "Please provide a city name"}), 400

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url)
        data = res.json()
        if data.get("cod") != 200:
            return jsonify({"error": data.get("message", "City not found")}), 404
        weather_info = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }
        return jsonify(weather_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ Book Routes (Open Library) ------------------ #
@app.route("/get_books", methods=["POST"])
def get_books():
    query = request.form.get("query", "crop farming")
    url = f"https://openlibrary.org/search.json?q={query}&limit=12"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        books = []
        for doc in data.get("docs", []):
            books.append({
                "title": doc.get("title"),
                "author": ", ".join(doc.get("author_name", [])),
                "first_publish_year": doc.get("first_publish_year"),
                "read_url": f"https://openlibrary.org{doc.get('key')}?embed=true"
            })
        return jsonify(books)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import json

# ------------------ Crop Routes ------------------ #
@app.route("/get_crops", methods=["GET"])
def get_crops():
    # Load the static crops.json file
    try:
        with open("crops.json", "r") as f:
            crops = json.load(f)
        return jsonify(crops)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/crop_cards")
def crop_cards_page():
    return render_template("crop_cards.html")


# ------------------ Optional Resources Page ------------------ #
@app.route("/resources")
def resources_page():
    return render_template("resources.html")


# ------------------ Run Server ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
