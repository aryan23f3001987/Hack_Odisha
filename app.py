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
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

if not HF_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY is not set in your .env file!")

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
# @app.route("/get_books", methods=["POST"])
# def get_books():
#     query = request.form.get("query", "crop farming")
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
#                 "read_url": f"https://openlibrary.org{doc.get('key')}?embed=true"
#             })
#         return jsonify(books)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


import json

@app.route("/get_books", methods=["POST"])
def get_books():
    query = request.form.get("query", "crop farming")
    url = f"https://openlibrary.org/search.json?q={query}&limit=12"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        books = []
        for doc in data.get("docs", []):
            # Build cover URL if cover_i exists, else use placeholder
            cover_url = f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg" if doc.get('cover_i') else '/static/placeholder.jpg'

            books.append({
                "title": doc.get("title"),
                "author": ", ".join(doc.get("author_name", [])),
                "first_publish_year": doc.get("first_publish_year"),
                "read_url": f"https://openlibrary.org{doc.get('key')}?embed=true",
                "cover_url": cover_url
            })
        return jsonify(books)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ------------------ YouTube Routes ------------------ #
import re
@app.route("/get_videos", methods=["POST"])
def get_videos():
    user_query = request.form.get("query", "")
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        return jsonify({"error": "Missing YOUTUBE_API_KEY"}), 500

    # Single search.list call for medium videos only
    search_url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults=50&q={user_query}&videoDuration=medium&key={api_key}"
    )
    search_res = requests.get(search_url, timeout=5).json()

    videos = []
    for item in search_res.get("items", []):
        videos.append({
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
            "video_url": f"https://www.youtube.com/embed/{item['id']['videoId']}"
        })

    # Limit to 12 videos for your frontend
    videos = videos[:12]

    return jsonify(videos)

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

@app.route("/detect_disease", methods=["POST"])
def detect_disease():
    try:
        # Get uploaded file and crop name
        image = request.files.get("image")
        crop_name = request.form.get("crop")

        if not image or not crop_name:
            return jsonify({"error": "Please provide both an image and a crop name"}), 400

        # Read image as bytes
        image_bytes = image.read()

        # Send POST request to Hugging Face
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        res = requests.post(HF_API_URL, headers=headers, data=image_bytes)

        if res.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {res.text}"}), res.status_code

        result = res.json()
        if not isinstance(result, list) or len(result) == 0:
            return jsonify({"error": "No prediction returned"}), 500

        # Get top class and remove crop prefix if needed
        top_class = result[0]["label"]
        disease_name = top_class.split("with")[-1].strip() if "with" in top_class else top_class

        # Prepare hidden prompt for chatbot
        hidden_prompt = f"Crop: {crop_name}\nDisease: {disease_name}\n\nGive a short treatment advice."
        response = chatModel.invoke(hidden_prompt)
        advice = response.content

        # Return top 5 predictions as well
        top5 = [{"label": r["label"], "score": r["score"]} for r in sorted(result, key=lambda x: x["score"], reverse=True)[:5]]

        return jsonify({
            "crop": crop_name,
            "disease": disease_name,
            "advice": advice,
            "top5_predictions": top5
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ Optional Resources Page ------------------ #
@app.route("/resources")
def resources_page():
    return render_template("resources.html")

# ------------------ Run Server ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
