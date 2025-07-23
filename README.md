BhashaSetu MVC Demo
Overview
BhashaSetu is a multilingual content translation and processing system designed to translate English text, audio, and video into multiple Indian languages. It leverages advanced machine learning models, including IndicTrans2 for text translation and Whisper for audio transcription, to provide accurate translations and generate audio, subtitles, and dubbed videos. This demo showcases the capabilities of the system through a web-based interface and a FastAPI backend.
The project supports translation into 12 Indian languages: Hindi, Bengali, Tamil, Telugu, Malayalam, Gujarati, Marathi, Punjabi, Kannada, Urdu, Odia, and Assamese. It includes features for translating PIB (Press Information Bureau) articles, audio files, and videos, as well as generating subtitles and dubbed videos.
Features

Text Translation: Translates English text into supported Indian languages with optional audio output using gTTS.
Audio Translation: Transcribes audio files using Whisper and translates the transcribed text into the target language.
Video Processing: Generates subtitles for videos and creates dubbed videos with translated audio.
PIB Article Translation: Fetches and translates PIB press releases from provided URLs, preserving HTML structure and generating audio for the translated content.
Performance Metrics: Provides detailed metrics (latency, cost, BLEU/WER scores) for each translation task.
Optimized Performance: Uses batch processing, parallelization, model quantization, and caching to reduce translation time.
Web Interface: A user-friendly front-end (index.html) for interacting with the API, displaying results, and downloading outputs.

Prerequisites
To run the BhashaSetu MVC Demo, ensure you have the following installed:

Python 3.8+
Dependencies (listed in requirements.txt):fastapi
uvicorn
transformers
torch
whisper
gtts
pydub
ffmpeg-python
requests
beautifulsoup4
nltk


FFmpeg: Required for audio and video processing. Install it on your system:
Ubuntu: sudo apt-get install ffmpeg
macOS: brew install ffmpeg
Windows: Download from FFmpeg website and add to PATH.


Hardware (recommended):
GPU (e.g., NVIDIA CUDA-enabled) for faster model inference.
At least 8GB RAM and 4 CPU cores for parallel processing.



Installation

Clone the Repository:
git clone https://github.com/your-repo/bhashasetu-mvc-demo.git
cd bhashasetu-mvc-demo


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download NLTK Data:
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


Install FFmpeg:Ensure FFmpeg is installed and accessible in your system PATH.


Usage
Running the Backend

Start the FastAPI Server:
uvicorn main:app --host 127.0.0.1 --port 8000


The server will be available at http://127.0.0.1:8000.


Accessing the Front-End

Open index.html in a web browser (e.g., Chrome, Firefox).

Ensure the backend server is running, as the front-end makes API calls to http://127.0.0.1:8000.
Alternatively, serve index.html using a simple HTTP server:python -m http.server 8080

Then access it at http://localhost:8080.


Use the interface to:

Translate PIB Articles: Enter a PIB URL (e.g., https://www.pib.gov.in/PressReleasePage.aspx?PRID=1942412) and select a target language.
Translate Audio: Upload an audio file and select a target language.
Generate Subtitles: Upload a video file to generate translated .srt subtitles.
Translate & Dub Video: Upload a video file to generate a dubbed video with translated audio.



API Endpoints
The backend provides the following endpoints:

GET /: Root endpoint to check if the API is running.
POST /translate/text/{target_lang}: Translates text into the specified Indian language.
POST /translate/audio/{target_lang}: Transcribes and translates an audio file.
POST /generate_subtitle/{target_lang}: Generates translated subtitles for a video.
POST /video_translate/{target_lang}: Translates and dubs a video.
GET /demo_pib_translation: Fetches and translates a PIB article from a URL.

Example API call (using curl):
curl -X POST "http://127.0.0.1:8000/translate/text/hi" -H "Content-Type: application/json" -d '{"text": "Hello, world!", "include_audio": true}'

Optimizations
The project has been optimized to reduce translation latency, particularly for large inputs like PIB articles:

Batch Processing: Processes text chunks in smaller batches (batch_size=32) to optimize GPU/CPU usage.
Parallelization: Uses ThreadPoolExecutor with 4 workers to parallelize translation and audio generation.
Model Quantization: Applies dynamic quantization on CPU to reduce model size and inference time.
Translation Caching: Caches translations for repeated text segments to avoid redundant computation.
Preprocessing Caching: Uses functools.lru_cache to cache preprocessing results.
HTML Filtering: Skips short or irrelevant text nodes (min_length=5) to reduce translation overhead.
Mixed Precision: Uses torch.cuda.amp for faster GPU inference.

These optimizations reduce translation time by an estimated 2.7-5.4x and audio generation time by 2-3x compared to the original implementation.
Performance Metrics
For a sample PIB article (~7774 characters):

Original:
Total Latency: ~643.67s
Translation Time: ~543.94s
Audio Generation: ~99s
Cost: ₹1.07279


Optimized (Estimated):
Total Latency: ~150-250s
Translation Time: ~100-200s
Audio Generation: ~33-50s
Cost: ~₹0.25-0.42



Project Structure
bhashasetu-mvc-demo/
├── main.py            # FastAPI backend with translation logic
├── index.html         # Front-end interface
├── requirements.txt   # Python dependencies
├── README.md          # This file

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

Please ensure code follows PEP 8 standards and include tests for new features.
Troubleshooting

Backend Errors:
Ensure the FastAPI server is running (uvicorn main:app --host 127.0.0.1 --port 8000).
Check for missing dependencies (pip install -r requirements.txt).
Verify FFmpeg is installed and accessible in PATH.


Performance Issues:
Monitor GPU/CPU usage with nvidia-smi or psutil.
Adjust batch_size and max_workers in main.py based on hardware.
Profile with cProfile to identify bottlenecks:python -m cProfile -s time main.py




Translation Quality:
Validate BLEU/WER scores in metrics to ensure translation accuracy.
Test with different inputs to confirm robustness.



License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
AI4Bharat: For the IndicTrans2 model.
OpenAI: For the Whisper transcription model.
Google: For the gTTS text-to-speech library.