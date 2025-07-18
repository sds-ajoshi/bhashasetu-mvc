from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import time
import torch
import whisper
import os
import base64
import io
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import ffmpeg
import requests
from bs4 import BeautifulSoup
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
import nltk
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import concurrent.futures
import re
nltk.download('punkt')
nltk.download('punkt_tab')

# --- Pydantic Models ---
class Metrics(BaseModel):
    latency_ms: float
    cost_rupees: float
    time_complexity: str
    space_complexity: str
    bleu_score: float | None = None
    wer_score: float | None = None

class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    target_language: str
    metrics: Metrics
    audio_base64: str | None = None

class AudioTranslationResponse(BaseModel):
    transcribed_text: str
    translation: TranslationResponse

class SubtitleResponse(BaseModel):
    srt_content: str
    translation: TranslationResponse

class VideoTranslationResponse(BaseModel):
    transcribed_text: str
    translation: TranslationResponse
    video_base64: str | None = None

class TranslationRequest(BaseModel):
    text: str
    ground_truth_text: str | None = None
    include_audio: bool = True

class PIBTranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    target_language: str
    metrics: Metrics
    audio_base64: str | None = None

app = FastAPI(title="BhashaSetu MVC")

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Supported Languages ---
LANGUAGES = {
    "hi": "hin_Deva",  # Hindi
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "bn": "ben_Beng",  # Bengali
    "ml": "mal_Mlym",  # Malayalam
    "gu": "guj_Gujr",  # Gujarati
    "mr": "mar_Deva",  # Marathi
    "pa": "pan_Guru",  # Punjabi
    "kn": "kan_Knda",  # Kannada
    "ur": "urd_Arab",  # Urdu
    "or": "ori_Orya",  # Odia
    "as": "asm_Beng"   # Assamese
}

# --- Model and Toolkit Initialization ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
print(f"Loading IndicTrans2 Model: {MODEL_NAME}")
translator_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
ip = IndicProcessor(inference=True)
print("IndicTrans2 loaded successfully.")

print("Loading Whisper Model (base)...")
whisper_model = whisper.load_model("base", device=DEVICE)
print("Whisper model loaded successfully.")

# Optional Coqui TTS setup (commented; requires pip install TTS and model download)
# try:
#     from TTS.api import TTS
#     COQUI_TTS = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
#     COQUI_AVAILABLE = True
# except ImportError:
#     COQUI_AVAILABLE = False

# --- Helper Functions ---
def calculate_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference)
    hyp_tokens = nltk.word_tokenize(hypothesis)
    return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))

def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return edit_distance(ref_words, hyp_words) / len(ref_words) if ref_words else 0.0

def extract_tables(text: str):
    """Extract table-like sections (e.g., annexures) using regex for simple tabular data."""
    # Simple regex to detect table blocks (e.g., lines with multiple \n separated columns)
    table_pattern = r'(Annexure\s+[I|II].*?)(?=Annexure\s+[I|II]|This information|\Z)'
    tables = re.findall(table_pattern, text, re.DOTALL | re.IGNORECASE)
    non_table_text = re.sub(table_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    return non_table_text, tables

def translate_table(table_text: str, src_lang: str, tgt_lang_full: str):
    """Translate table by splitting into rows/cells, translate each, reconstruct."""
    lines = table_text.split('\n')
    translated_lines = []
    for line in lines:
        if line.strip():
            # Assume tab-like: split by multiple spaces or \n
            cells = re.split(r'\s{2,}|\t', line.strip())
            translated_cells = []
            for cell in cells:
                if cell:
                    batch = ip.preprocess_batch([cell], src_lang=src_lang, tgt_lang=tgt_lang_full)
                    inputs = translator_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
                    with torch.no_grad():
                        generated_tokens = translator_model.generate(**inputs, use_cache=True, min_length=0, max_length=128, num_beams=3, num_return_sequences=1)
                    decoded_tokens = translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    translations = ip.postprocess_batch(decoded_tokens, lang=tgt_lang_full)
                    translated_cells.append(translations[0])
            translated_lines.append('\t'.join(translated_cells))  # Reconstruct as tab-separated
    return '\n'.join(translated_lines)

def translate_chunk(chunk: str, src_lang: str, tgt_lang_full: str):
    batch = ip.preprocess_batch([chunk], src_lang=src_lang, tgt_lang=tgt_lang_full)
    inputs = translator_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
    with torch.no_grad():
        generated_tokens = translator_model.generate(**inputs, use_cache=True, min_length=0, max_length=512, num_beams=3, num_return_sequences=1)
    decoded_tokens = translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    translations = ip.postprocess_batch(decoded_tokens, lang=tgt_lang_full)
    return translations[0]

def translate_text(text: str, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True):
    if not text:
        return "", Metrics(latency_ms=0.0, cost_rupees=0.0, time_complexity="O(1)", space_complexity="O(1)"), None

    start_time = time.time()
    try:
        # Extract and handle tables separately
        non_table_text, tables = extract_tables(text)
        
        # Chunk non-table text into batches of 5-10 sentences
        sentences = nltk.sent_tokenize(non_table_text)
        chunk_size = 8  # Batch 8 sentences per generation for efficiency
        translated_sentences = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Parallelize chunks
            futures = []
            for i in range(0, len(sentences), chunk_size):
                chunk = ' '.join(sentences[i:i+chunk_size])
                futures.append(executor.submit(translate_chunk, chunk, src_lang, tgt_lang_full))
            for future in concurrent.futures.as_completed(futures):
                translated_sentences.append(future.result())
        
        translated_non_table = ' '.join(translated_sentences)
        
        # Translate tables
        translated_tables = []
        for table in tables:
            translated_tables.append(translate_table(table, src_lang, tgt_lang_full))
        
        # Reassemble
        translated_text = translated_non_table + '\n\n' + '\n\n'.join(translated_tables)
    except Exception as e:
        print(f"Translation error: {e}")
        raise ValueError("Translation failed")

    latency = round((time.time() - start_time) * 1000, 3)  # in ms
    cost = round((latency / 60000) * 0.10, 5)
    
    # Quality check: If ground_truth, split into chunks and average BLEU
    bleu_score = None
    if ground_truth:
        gt_sentences = nltk.sent_tokenize(ground_truth)
        hyp_sentences = nltk.sent_tokenize(translated_text)
        chunk_bleus = [calculate_bleu(' '.join(gt_sentences[i:i+chunk_size]), ' '.join(hyp_sentences[i:i+chunk_size])) for i in range(0, min(len(gt_sentences), len(hyp_sentences)), chunk_size)]
        bleu_score = sum(chunk_bleus) / len(chunk_bleus) if chunk_bleus else None

    metrics_data = Metrics(
        latency_ms=latency,
        cost_rupees=cost,
        time_complexity="O(n^2 * d)",
        space_complexity="O(n * d)",
        bleu_score=bleu_score
    )

    audio_base64 = None
    if include_audio:
        try:
            tts = gTTS(translated_text, lang=tgt_lang_short)
            buffer = io.BytesIO()
            tts.save(buffer)
            buffer.seek(0)
            audio_bytes = buffer.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"gTTS generation error: {e}")
            # Optional: Fallback to Coqui TTS if available
            # if COQUI_AVAILABLE:
            #     try:
            #         wav = COQUI_TTS.tts(text=translated_text, language=tgt_lang_short)  # Adjust language if needed
            #         # Convert wav to mp3 if necessary, then base64
            #     except Exception as ce:
            #         print(f"Coqui TTS error: {ce}")

    return translated_text, metrics_data, audio_base64

def translate_with_timeout(text: str, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True, timeout_sec: int = 60):
    """Wrapper for translate_text with timeout using concurrent.futures."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(translate_text, text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            return text, Metrics(latency_ms=timeout_sec*1000, cost_rupees=0.0, time_complexity="O(1)", space_complexity="O(1)"), None  # Return partial/original on timeout

async def translate_audio_file(audio_file: UploadFile, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True):
    temp_audio_path = f"./temp_{audio_file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        print("Transcribing audio...")
        try:
            transcription_result = whisper_model.transcribe(temp_audio_path, fp16=torch.cuda.is_available(), language="en")
            transcribed_text = transcription_result["text"]
            wer_score = calculate_wer(ground_truth, transcribed_text) if ground_truth else None
        except Exception as e:
            print(f"Transcription error: {e}")
            raise ValueError("Audio transcription failed")

        print("Translating transcribed text...")
        translated_text, metrics, audio_base64 = translate_with_timeout(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio)
        metrics.wer_score = wer_score

        translation_response = TranslationResponse(
            source_text=transcribed_text,
            translated_text=translated_text,
            target_language=tgt_lang_short,
            metrics=metrics,
            audio_base64=audio_base64
        )
        return AudioTranslationResponse(transcribed_text=transcribed_text, translation=translation_response)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

async def generate_subtitles(file: UploadFile, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True):
    temp_path = f"./temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        print("Transcribing with timestamps...")
        try:
            result = whisper_model.transcribe(temp_path, fp16=torch.cuda.is_available(), language="en", verbose=False)
            segments = result["segments"]
            transcribed_text = result["text"]
            wer_score = calculate_wer(ground_truth, transcribed_text) if ground_truth else None
        except Exception as e:
            print(f"Transcription error: {e}")
            raise ValueError("Transcription failed")

        print("Translating segments...")
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            if text:
                translated, _, _ = translate_with_timeout(text, src_lang, tgt_lang_full, tgt_lang_short, include_audio=False)
                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start-int(start))*1000):03d}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end-int(end))*1000):03d}"
                srt_content += f"{i}\n{start_time} --> {end_time}\n{translated}\n\n"

        translated_full, metrics, audio_base64 = translate_with_timeout(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio)
        metrics.wer_score = wer_score

        translation = TranslationResponse(
            source_text=transcribed_text,
            translated_text=translated_full,
            target_language=tgt_lang_short,
            metrics=metrics,
            audio_base64=audio_base64
        )
        return SubtitleResponse(srt_content=srt_content, translation=translation)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def video_translate(video_file: UploadFile, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True):
    temp_video_path = f"./temp_video_{video_file.filename}"
    temp_audio_path = f"./temp_audio_{video_file.filename}.mp3"
    temp_tts_path = f"./temp_tts_{video_file.filename}.mp3"
    temp_output_path = f"./temp_output_{video_file.filename}.mp4"
    try:
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await video_file.read())

        # Extract audio
        print("Extracting audio...")
        ffmpeg.input(temp_video_path).output(temp_audio_path, format='mp3', acodec='libmp3lame').run(overwrite_output=True, quiet=True, cmd='ffmpeg')

        # Transcribe
        print("Transcribing audio...")
        result = whisper_model.transcribe(temp_audio_path, fp16=torch.cuda.is_available(), language="en")
        transcribed_text = result["text"]
        wer_score = calculate_wer(ground_truth, transcribed_text) if ground_truth else None

        # Translate
        print("Translating text...")
        translated_text, metrics, _ = translate_with_timeout(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio=False)  # Audio generated separately
        metrics.wer_score = wer_score

        # Generate TTS audio file
        try:
            tts = gTTS(translated_text, lang=tgt_lang_short)
            tts.save(temp_tts_path)
        except Exception as e:
            print(f"TTS error: {e}")
            raise ValueError("TTS generation failed")

        # Overlay new audio on video
        print("Overlaying audio on video...")
        try:
            input_video = ffmpeg.input(temp_video_path)
            input_audio = ffmpeg.input(temp_tts_path)
            ffmpeg.output(input_video.video, input_audio.audio, temp_output_path, vcodec='copy', acodec='aac', strict='experimental').run(overwrite_output=True, quiet=True, cmd='ffmpeg')
        except Exception as e:
            print(f"FFmpeg error: {e}")
            raise ValueError("Video processing failed")

        # Read video as base64
        with open(temp_output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Optional audio_base64
        audio_base64 = None
        if include_audio:
            with open(temp_tts_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return VideoTranslationResponse(
            transcribed_text=transcribed_text,
            translation=TranslationResponse(
                source_text=transcribed_text,
                translated_text=translated_text,
                target_language=tgt_lang_short,
                metrics=metrics,
                audio_base64=audio_base64
            ),
            video_base64=video_base64
        )
    finally:
        for path in [temp_video_path, temp_audio_path, temp_tts_path, temp_output_path]:
            if os.path.exists(path):
                os.remove(path)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "BhashaSetu MVC API with Speech Translation is running!"}

@app.post("/translate/text/{target_lang}", response_model=TranslationResponse, tags=["Text Translation"])
def translate_text_endpoint(target_lang: str, request: TranslationRequest):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    translated_text, metrics, audio_base64 = translate_with_timeout(request.text, "eng_Latn", tgt_lang_full, target_lang, request.ground_truth_text, request.include_audio)
    return TranslationResponse(
        source_text=request.text,
        translated_text=translated_text,
        target_language=target_lang,
        metrics=metrics,
        audio_base64=audio_base64
    )

@app.post("/translate/audio/{target_lang}", response_model=AudioTranslationResponse, tags=["Audio Translation"])
async def translate_audio_endpoint(target_lang: str, audio_file: UploadFile = File(...), ground_truth_text: str | None = Query(None), include_audio: bool = Query(True)):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    return await translate_audio_file(audio_file, "eng_Latn", tgt_lang_full, target_lang, ground_truth_text, include_audio)

@app.post("/generate_subtitle/{target_lang}", response_model=SubtitleResponse, tags=["Subtitles"])
async def generate_subtitle_endpoint(target_lang: str, file: UploadFile = File(...), ground_truth_text: str | None = Query(None), include_audio: bool = Query(True)):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    return await generate_subtitles(file, "eng_Latn", tgt_lang_full, target_lang, ground_truth_text, include_audio)

@app.post("/video_translate/{target_lang}", response_model=VideoTranslationResponse, tags=["Video Translation"])
async def video_translate_endpoint(target_lang: str, video_file: UploadFile = File(...), ground_truth_text: str | None = Query(None), include_audio: bool = Query(True)):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    return await video_translate(video_file, "eng_Latn", tgt_lang_full, target_lang, ground_truth_text, include_audio)

@app.get("/demo_pib_translation", response_model=PIBTranslationResponse, tags=["Demo"])
def demo_pib_translation(url: str = Query(...), lang: str = Query("hi"), include_audio: bool = Query(True)):
    tgt_lang_full = LANGUAGES.get(lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    try:
        # Force English version by setting 'lang=1'
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        query_params['lang'] = ['1']  # Override to English
        new_query = urlencode(query_params, doseq=True)
        english_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, new_query, parsed_url.fragment))
        
        response = requests.get(english_url, timeout=10)  # Add timeout to prevent hanging
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='innner-page-main-about-us-content-right-part')
        if not content_div:
            raise ValueError("Content div not found")
        # Extract body text from all <p> tags within the div
        paragraphs = content_div.find_all('p')
        original_text = '\n\n'.join(p.get_text(separator='\n', strip=True) for p in paragraphs if p.get_text(strip=True))
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Request to PIB timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch or parse PIB article: {str(e)}")
    translated_text, metrics, audio_base64 = translate_with_timeout(original_text, "eng_Latn", tgt_lang_full, lang, include_audio=include_audio)
    return PIBTranslationResponse(
        original_text=original_text,
        translated_text=translated_text,
        target_language=lang,
        metrics=metrics,
        audio_base64=audio_base64
    )