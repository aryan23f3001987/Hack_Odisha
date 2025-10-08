# 🌾 CropWise - AI-Powered Farming Assistant

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg" alt="OpenAI">
  <img src="https://img.shields.io/badge/Pinecone-Vector--DB-purple.svg" alt="Pinecone">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

## 📖 Overview

CropWise is an intelligent farming assistant that leverages AI and machine learning to provide farmers with expert advice on crop management, pest control, weather insights, and sustainable farming practices. Built with Flask and powered by OpenAI's GPT-4o-mini, it uses Retrieval Augmented Generation (RAG) with Pinecone vector database to deliver accurate, context-aware agricultural guidance.

## ✨ Key Features

### 🤖 AI-Powered Chat Assistant
- Interactive chat interface with intelligent formatting
- Context-aware responses using RAG architecture
- Memory persistence for conversation continuity
- Smart content detection and styling for warnings, recommendations, and tips

### 🌦️ Advanced Weather Integration

- Real-time weather data using OpenWeatherMap API
- 5-day detailed forecast with hourly breakdowns
- Location-based weather using geocoding
- Comprehensive metrics: temperature, humidity, wind speed, rainfall, visibility
- Weather impact analysis for farming activities

### 📚 Educational Resources
- Integration with Open Library for agricultural books
- Curated educational content and tutorials
- Comprehensive farming guides and best practices
- YouTube video integration for farming tutorials

### 🌱 Crop Information Cards
- Detailed crop information database
- Growing seasons, water requirements, and temperature needs
- Cost analysis and yield predictions

### 🎨 Modern Dark Theme UI
- Responsive design with glassmorphism effects
- Smooth animations and intuitive navigation
- Mobile-optimized interface
- Accessibility-focused design

### 💰 Market Price Tracking

- Real-time agricultural commodity prices
- Integration with India's Agriculture Market API (data.gov.in)
- Filter by state, district, and crop type
- Daily mandi price updates with min/max/modal prices
- Historical price trends and analysis

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: OpenAI GPT-4o-mini, LangChain
- **Vector Database**: Pinecone
- **Embeddings**: OpenAI text-embedding-3-small
- **APIs**: OpenWeatherMap, Open Library
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with CSS Variables

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key
- Pinecone API Key
- OpenWeatherMap API Key
- Hugging Face API Key
- YouTube Data API Key
- Agriculture Market API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cropwise.git
   cd cropwise
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   WEATHER_API_KEY=your_openweathermap_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   YOUTUBE_API_KEY=your_youtube_api_key_here
   AGRICULTURE_MARKET_API_KEY=your_agriculture_market_api_key_here
   ```

5. **Prepare your agricultural data**
   
   Place your PDF files containing agricultural knowledge in the `data/` directory.

6. **Set up Pinecone Vector Store**
   ```bash
   python store_index.py
   ```
   This will:
   - Load and process your PDF files
   - Split documents into manageable chunks
   - Generate embeddings using OpenAI
   - Upload to Pinecone vector database

7. **Run the application**
   ```bash
   python app.py
   ```

8. **Access the application**
   
   Open your browser and navigate to `http://localhost:8080`

## 📁 Project Structure

```
cropwise/
├── app.py                 # Main Flask application
├── store_index.py         # Pinecone index setup and data upload
├── crops.json            # Static crop information database
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── data/                 # PDF files for knowledge base
├── sessions/             # Chat session storage (auto-created)
├── src/
│   ├── helper.py         # Utility functions for PDF processing
│   ├── prompt.py         # System prompts for AI assistant
│   └── chat_memory.py    # Chat memory management
└── templates/
    ├── login.html        # Login/authentication page
    ├── chat.html         # Main chat interface
    ├── weather.html      # Weather dashboard
    ├── crop_cards.html   # Crop information cards
    ├── MarketPrices.html # Market price tracking
    └── resources.html    # Educational resources
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | ✅ |
| `PINECONE_API_KEY` | Your Pinecone API key | ✅ |
| `WEATHER_API_KEY` | OpenWeatherMap API key | ✅ |
| `HUGGINGFACE_API_KEY` | Hugging Face API key for disease detection | ✅ |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key | ✅ |
| `AGRICULTURE_MARKET_API_KEY` | India Agriculture Market API key | ✅ |
| `PORT` | Server port (default: 8080) | ❌ |

### Customization Options

- **Chunk Size**: Modify `chunk_size` in `store_index.py` (default: 500)
- **Similarity Search**: Adjust `k` parameter in `app.py` (default: 3)
- **Chat Model**: Change model in `app.py` (default: gpt-4o-mini)
- **Embedding Model**: Modify in both files (default: text-embedding-3-small)
- **Disease Detection Model**: Change HF model URL in `app.py`
- **Confidence Threshold**: Adjust disease detection confidence threshold (default: 0.4)

## 📊 Features Deep Dive

### RAG (Retrieval Augmented Generation)
The application uses a sophisticated RAG pipeline:
1. **Document Processing**: PDFs are loaded and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings stored in Pinecone for fast similarity search
4. **Query Processing**: User queries are embedded and matched against stored knowledge
5. **Response Generation**: Relevant context is fed to GPT for accurate responses

### Intelligent Chat Memory
- Session-based memory storage using JSON files
- Conversation context preserved across interactions
- Automatic session cleanup on server restart

### Smart Content Detection
The UI automatically detects and styles different types of content:
- ⚠️ **Warnings/Precautions**: Red styling for safety information
- ✅ **Recommendations**: Green styling for best practices
- ℹ️ **Information/Tips**: Blue styling for helpful notes
- 🚨 **Dangers/Alerts**: Red alerts for critical information

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/get` | POST | Send chat messages |
| `/weather` | GET | Weather dashboard |
| `/get_weather` | POST | Get weather data |
| `/get_books` | POST | Search agricultural books |
| `/get_crops` | GET | Retrieve crop information |
| `/crop_cards` | GET | Crop cards interface |
| `/resources` | GET | Educational resources |

## 👥 Contributors

We're grateful to the following contributors who made CropWise possible:

| Developer | LinkedIn |
|-----------|----------|
| **Aryan Thakur** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aryan-thakur-b3075a2ba/) |
| **Shadan Zameer** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zameershadan/) |

</div>

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments for complex logic
- Test your changes thoroughly
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing powerful language models
- **Pinecone** for vector database services
- **OpenWeatherMap** for weather data
- **Open Library** for book recommendations
- **LangChain** for RAG implementation framework

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all API keys are correctly set in `.env` file
   - Verify API keys have sufficient credits/permissions

2. **Pinecone Connection Issues**
   - Check Pinecone API key and region settings
   - Ensure index name matches in both `store_index.py` and `app.py`

3. **Memory Issues**
   - Reduce `chunk_size` if encountering memory problems
   - Process PDFs in smaller batches

4. **Empty Responses**
   - Verify Pinecone index contains data
   - Check if PDF files are properly formatted and readable

### Getting Help

- 📧 Open an issue on GitHub
- 💬 Check existing issues for solutions
- 📖 Review the documentation carefully

## 🔮 Future Enhancements

- **Image-based Crop Disease Detection:**  
  Farmers will be able to upload images of their crops, and the AI-powered bot will detect diseases, nutrient deficiencies, and suggest actionable solutions to improve yield and plant health.

- **Mobile App Development:**  
  Develop a mobile application for easy access to crop insights, weather updates, and personalized guidance anytime, anywhere.

- **IoT Sensor Integration:**  
  Use IoT sensors to monitor soil moisture, temperature, and other environmental conditions, enabling precision irrigation and automated alerts.

- **Community & Marketplace Features:**  
  Build a platform for farmers to connect, share knowledge, ask questions, and access a marketplace for seeds, fertilizers, and equipment.

---

<div align="center">
  <p><strong>CropWise</strong> - Empowering farmers with AI-driven agricultural insights 🌾</p>
  <p>Made with ❤️ for the farming community</p>
</div>