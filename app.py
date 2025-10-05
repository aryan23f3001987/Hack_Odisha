# ------------------ Actual App ------------------ #
from flask import Flask, render_template, jsonify, request, redirect, url_for
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
import json
from datetime import datetime
import firebase_admin

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
@app.route("/login")
def login_page():
    return render_template("login.html")

# This route remains for the chat page itself.
@app.route("/chat")
def chat_page():
    return render_template("chat.html")

# ------------------ Chat API ------------------ #
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

# ------------------ Weather API ------------------ #
# Add this import at the top of your file if it's not already there
from datetime import datetime

@app.route("/weather")
def weather_page():
    return render_template("weather.html")

@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.form.get("city")
    if not city:
        return jsonify({"error": "City name is required"}), 400

    try:
        # Step 1: Convert city name to latitude and longitude (Geocoding)
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={WEATHER_API_KEY}"
        geo_res = requests.get(geo_url)
        geo_res.raise_for_status()
        geo_data = geo_res.json()

        if not geo_data:
            return jsonify({"error": f"City '{city}' not found."}), 404

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city_name = geo_data[0]["name"]

        # Step 2: Get current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        current_res = requests.get(current_url)
        current_res.raise_for_status()
        current_data = current_res.json()

        current_weather = {
            "city": city_name,
            "temperature": current_data["main"]["temp"],
            "feels_like": current_data["main"]["feels_like"],
            "humidity": current_data["main"]["humidity"],
            "wind_speed": current_data["wind"]["speed"],
            "wind_deg": current_data["wind"].get("deg", 0),
            "visibility": current_data.get("visibility", 0),
            "description": current_data["weather"][0]["description"]
        }

        # Step 3: Get 5-day / 3-hour forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        forecast_res = requests.get(forecast_url)
        forecast_res.raise_for_status()
        forecast_data = forecast_res.json()

        # Aggregate to daily forecast (next 5 days)
        daily_forecast = {}
        for item in forecast_data["list"]:
            date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
            temp_max = item["main"]["temp_max"]
            temp_min = item["main"]["temp_min"]
            desc = item["weather"][0]["description"]
            humidity = item["main"]["humidity"]
            wind_speed = item["wind"]["speed"]
            rain = item.get("rain", {}).get("3h", 0.0)

            if date not in daily_forecast:
                daily_forecast[date] = {
                    "max_temp": temp_max,
                    "min_temp": temp_min,
                    "description": desc,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "rain": rain
                }
            else:
                daily_forecast[date]["max_temp"] = max(daily_forecast[date]["max_temp"], temp_max)
                daily_forecast[date]["min_temp"] = min(daily_forecast[date]["min_temp"], temp_min)
                daily_forecast[date]["rain"] += rain  # sum rainfall

        # Take next 5 days
        forecast_list = []
        for i, (date, info) in enumerate(daily_forecast.items()):
            if i >= 5:
                break
            forecast_list.append({
                "day": datetime.strptime(date, "%Y-%m-%d").strftime('%a'),
                "description": info["description"],
                "max_temp": info["max_temp"],
                "min_temp": info["min_temp"],
                "humidity": info["humidity"],
                "wind_speed": info["wind_speed"],
                "rain": info["rain"]
            })

        final_response = {
            "current": current_weather,
            "forecast": forecast_list
        }

        return jsonify(final_response)

    except requests.exceptions.HTTPError as http_err:
        return jsonify({"error": f"API request failed: {http_err.response.reason}"}), http_err.response.status_code
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# ------------------ Book API ------------------ #l
@app.route("/get_books", methods=["POST"])
def get_books():
    query = request.form.get("query", "crop farming")
    url = f"https://openlibrary.org/search.json?q={query}&limit=12"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        books = []
        for doc in data.get("docs", []):
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

# ------------------ YouTube API ------------------ #
@app.route("/get_videos", methods=["POST"])
def get_videos():
    user_query = request.form.get("query", "")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing YOUTUBE_API_KEY"}), 500

    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=50&q={user_query}&videoDuration=medium&key={api_key}"
    search_res = requests.get(search_url, timeout=5).json()

    videos = []
    for item in search_res.get("items", []):
        videos.append({
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
            "video_url": f"https://www.youtube.com/embed/{item['id']['videoId']}"
        })

    return jsonify(videos[:12])

# ------------------ Crops API ------------------ #
@app.route("/get_crops", methods=["GET"])
def get_crops():
    try:
        with open("crops.json", "r") as f:
            crops = json.load(f)
        return jsonify(crops)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/crop_cards")
def crop_cards_page():
    return render_template("crop_cards.html")

# ------------------ Disease Detection ------------------ #
@app.route("/detect_disease", methods=["POST"])
def detect_disease():
    try:
        image = request.files.get("image")
        crop_name = request.form.get("crop")

        if not image or not crop_name:
            return jsonify({"error": "Please provide both an image and a crop name"}), 400

        # Read image
        image_bytes = image.read()
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

        # Get top prediction (always)
        top_prediction = max(result, key=lambda x: x["score"])

        # If confidence is too low (<0.3), return null disease
        if top_prediction["score"] < 0.4:
            advice = "Can't confidently guess the disease. Maybe try uploading a clearer picture."
            top5 = [
                {"label": r["label"], "score": r["score"]}
                for r in sorted(result, key=lambda x: x["score"], reverse=True)[:5]
            ]
            return jsonify({
                "crop": crop_name,
                "disease": None,
                "confidence": top_prediction["score"],
                "advice": advice,
                "top5_predictions": top5
            })

        # Otherwise return disease + advice
        disease_name = (
            top_prediction["label"].split("with")[-1].strip()
            if "with" in top_prediction["label"]
            else top_prediction["label"]
        )

        hidden_prompt = f"Crop: {crop_name}\nDisease: {disease_name}\n\nGive a short treatment advice."
        response = chatModel.invoke(hidden_prompt)
        advice = response.content

        top5 = [
            {"label": r["label"], "score": r["score"]}
            for r in sorted(result, key=lambda x: x["score"], reverse=True)[:5]
        ]

        return jsonify({
            "crop": crop_name,
            "disease": disease_name,
            "confidence": top_prediction["score"],
            "advice": advice,
            "top5_predictions": top5
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ------------------ Market Prices ------------------ #
@app.route("/market_prices")
def market_prices_page():
    return render_template("MarketPrices.html")


@app.route("/get_market_prices", methods=["GET"])
def get_market_prices():
    api_key = os.getenv("AGRICULTURE_MARKET_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing AGRICULTURE_MARKET_API_KEY"}), 500

    # Example: data.gov.in resource for daily mandi prices
    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"  # Replace with the real resource ID
    crops = request.args.getlist("crops")  # optional ?crops=Wheat&crops=Rice
    state = request.args.get("state", "")  # optional filter
    limit = request.args.get("limit", "100")

    # Build URL
    url = f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json&limit={limit}"
    if state:
        url += f"&filters[state]={state}"

    try:
        res = requests.get(url, timeout=10)
        data = res.json()

        if "records" not in data:
            return jsonify({"error": "No records found"}), 404

        records = data["records"]

        # Filter for selected crops (if provided)
        if crops:
            records = [r for r in records if r.get("commodity") in crops]

        # Prepare structured data
        # Example record fields: commodity, market, state, district, min_price, max_price, modal_price, arrival_date
        formatted = []
        for r in records:
            formatted.append({
                "commodity": r.get("commodity"),
                "market": r.get("market"),
                "state": r.get("state"),
                "district": r.get("district"),
                "date": r.get("arrival_date"),
                "modal_price": float(r.get("modal_price", 0)),
                "min_price": float(r.get("min_price", 0)),
                "max_price": float(r.get("max_price", 0))
            })

        return jsonify(formatted)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ Resources ------------------ #
@app.route("/resources")
def resources_page():
    return render_template("resources.html")

# ------------------ Run Server ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)