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
import logging
import subprocess
nltk.download('punkt')
nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class Metrics(BaseModel):
    latency_ms: float
    cost_rupees: float
    time_complexity: str
    space_complexity: str
    fetch_ms: float
    parse_ms: float
    translate_ms: float
    bleu_score: float | None = None
    wer_score: float | None = None

class TranslationResponse(BaseModel):
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    char_count: int
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
    source_language: str
    target_language: str
    original_text: str
    translated_text: str
    char_count: int
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
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        logger.info(f"GPU available: \n{nvidia_smi}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("nvidia-smi failed; GPU may not be properly configured")

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
logger.info(f"Loading IndicTrans2 Model: {MODEL_NAME}")
translator_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
ip = IndicProcessor(inference=True)
logger.info("IndicTrans2 loaded successfully.")

logger.info("Loading Whisper Model (base)...")
whisper_model = whisper.load_model("base", device=DEVICE)
logger.info("Whisper model loaded successfully.")

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
    table_pattern = r'(Annexure\s+[IVXLC]+.*?)(?=Annexure\s+[IVXLC]+|This information|\Z)'
    tables = re.findall(table_pattern, text, re.DOTALL | re.IGNORECASE)
    non_table_text = re.sub(table_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    return non_table_text, tables

def translate_chunks(chunks: list[str], src_lang: str, tgt_lang_full: str):
    if not chunks: return []
    logger.info(f"Translating {len(chunks)} chunks...")
    try:
        batch = ip.preprocess_batch(chunks, src_lang=src_lang, tgt_lang=tgt_lang_full)
        inputs = translator_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            generated_tokens = translator_model.generate(**inputs, use_cache=True, min_length=0, max_length=512, num_beams=1, num_return_sequences=1)
        decoded_tokens = translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = ip.postprocess_batch(decoded_tokens, lang=tgt_lang_full)
        return translations
    except Exception as e:
        logger.error(f"Error in translate_chunks: {e}")
        return [""] * len(chunks)

def translate_text(text: str, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True):
    if not text:
        return "", Metrics(latency_ms=0.0, cost_rupees=0.0, time_complexity="O(1)", space_complexity="O(1)", fetch_ms=0.0, parse_ms=0.0, translate_ms=0.0), None

    start_time = time.time()
    translated_text = ""
    try:
        non_table_text, tables = extract_tables(text)
        sentences = nltk.sent_tokenize(non_table_text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        for sent in sentences:
            word_count = len(nltk.word_tokenize(sent))
            if current_len + word_count > 200:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_len = word_count
            else:
                current_chunk.append(sent)
                current_len += word_count
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        translated_chunks = translate_chunks(chunks, src_lang, tgt_lang_full)
        translated_text = ' '.join(translated_chunks)

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise ValueError(f"Translation failed: {e}")

    translate_time = round((time.time() - start_time) * 1000, 3)
    cost = round((translate_time / 60000) * 0.10, 5)
    bleu_score = calculate_bleu(ground_truth, translated_text) if ground_truth else None
    
    metrics_data = Metrics(
        latency_ms=translate_time, cost_rupees=cost,
        time_complexity="O(n^2 * d)", space_complexity="O(n * d)",
        fetch_ms=0.0, parse_ms=0.0, translate_ms=translate_time,
        bleu_score=bleu_score
    )

    audio_base64 = None
    if include_audio:
        try:
            tts = gTTS(translated_text, lang=tgt_lang_short)
            buf = io.BytesIO()
            tts.save(buf)
            buf.seek(0)
            audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"gTTS error: {e}")

    return translated_text, metrics_data, audio_base64

async def generate_subtitles(file: UploadFile, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None):
    temp_path = f"./temp_subtitle_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        logger.info("Transcribing with timestamps for subtitles...")
        result = whisper_model.transcribe(temp_path, fp16=torch.cuda.is_available(), language="en")
        segments = result["segments"]
        transcribed_text = result["text"]
        wer_score = calculate_wer(ground_truth, transcribed_text) if ground_truth else None
        
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            if text:
                translated_segment, _, _ = translate_text(text, src_lang, tgt_lang_full, tgt_lang_short, include_audio=False)
                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start-int(start))*1000):03d}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end-int(end))*1000):03d}"
                srt_content += f"{i}\n{start_time} --> {end_time}\n{translated_segment}\n\n"

        translated_full, metrics, audio_base64 = translate_text(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio=True)
        metrics.wer_score = wer_score

        translation = TranslationResponse(
            source_language="en", target_language=tgt_lang_short,
            source_text=transcribed_text, translated_text=translated_full,
            char_count=len(transcribed_text), metrics=metrics, audio_base64=audio_base64
        )
        return SubtitleResponse(srt_content=srt_content, translation=translation)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def video_translate(video_file: UploadFile, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None):
    # Define temporary file paths
    base_name = os.path.splitext(video_file.filename)[0]
    temp_video_path = f"./temp_{base_name}.mp4"
    temp_audio_path = f"./temp_{base_name}.mp3"
    temp_tts_path = f"./temp_tts_{base_name}.mp3"
    temp_output_path = f"./temp_output_{base_name}.mp4"
    
    try:
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await video_file.read())

        logger.info("Extracting audio from video...")
        ffmpeg.input(temp_video_path).output(temp_audio_path, acodec='libmp3lame').run(overwrite_output=True, quiet=True)

        logger.info("Transcribing audio...")
        result = whisper_model.transcribe(temp_audio_path, fp16=torch.cuda.is_available(), language="en")
        transcribed_text = result["text"]
        wer_score = calculate_wer(ground_truth, transcribed_text) if ground_truth else None

        logger.info("Translating text...")
        translated_text, metrics, audio_base64 = translate_text(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio=True)
        metrics.wer_score = wer_score

        logger.info("Saving TTS audio to file...")
        if audio_base64:
            tts_audio_bytes = base64.b64decode(audio_base64)
            with open(temp_tts_path, "wb") as f:
                f.write(tts_audio_bytes)
        else:
            raise ValueError("TTS generation failed, cannot proceed with video dubbing.")

        logger.info("Overlaying new audio onto video...")
        input_video = ffmpeg.input(temp_video_path)
        input_audio = ffmpeg.input(temp_tts_path)
        ffmpeg.output(input_video.video, input_audio.audio, temp_output_path, vcodec='copy', acodec='aac', shortest=None).run(overwrite_output=True, quiet=True)

        logger.info("Encoding final video to base64...")
        with open(temp_output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        translation_response = TranslationResponse(
            source_language="en", target_language=tgt_lang_short,
            source_text=transcribed_text, translated_text=translated_text,
            char_count=len(transcribed_text), metrics=metrics, audio_base64=audio_base64
        )
        return VideoTranslationResponse(
            transcribed_text=transcribed_text,
            translation=translation_response,
            video_base64=video_base64
        )
    finally:
        for path in [temp_video_path, temp_audio_path, temp_tts_path, temp_output_path]:
            if os.path.exists(path):
                os.remove(path)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "BhashaSetu MVC API is running!"}

@app.post("/translate/text/{target_lang}", response_model=TranslationResponse, tags=["Text Translation"])
def translate_text_endpoint(target_lang: str, request: TranslationRequest):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")

    translated_text, metrics, audio_base64 = translate_text(
        text=request.text, src_lang="eng_Latn",
        tgt_lang_full=tgt_lang_full, tgt_lang_short=target_lang,
        ground_truth=request.ground_truth_text, include_audio=request.include_audio
    )
    
    return TranslationResponse(
        source_language="en", target_language=target_lang,
        source_text=request.text, translated_text=translated_text,
        char_count=len(request.text), metrics=metrics, audio_base64=audio_base64
    )

@app.post("/translate/audio/{target_lang}", response_model=AudioTranslationResponse, tags=["Audio Translation"])
async def translate_audio_endpoint(target_lang: str, audio_file: UploadFile = File(...), ground_truth_text: str | None = Query(None), include_audio: bool = Query(True)):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    
    temp_audio_path = f"./temp_{audio_file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        transcription_result = whisper_model.transcribe(temp_audio_path, fp16=torch.cuda.is_available(), language="en")
        transcribed_text = transcription_result["text"]
        wer_score = calculate_wer(ground_truth_text, transcribed_text) if ground_truth_text else None
        
        translated_text, metrics, audio_base64 = translate_text(transcribed_text, "eng_Latn", tgt_lang_full, target_lang, ground_truth_text, include_audio)
        metrics.wer_score = wer_score

        translation_response = TranslationResponse(
            source_language="en", target_language=target_lang,
            source_text=transcribed_text, translated_text=translated_text,
            char_count=len(transcribed_text), metrics=metrics, audio_base64=audio_base64
        )
        return AudioTranslationResponse(transcribed_text=transcribed_text, translation=translation_response)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/generate_subtitle/{target_lang}", response_model=SubtitleResponse, tags=["Subtitles"])
async def generate_subtitle_endpoint(target_lang: str, file: UploadFile = File(...), ground_truth_text: str | None = Query(None)):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    return await generate_subtitles(file, "eng_Latn", tgt_lang_full, target_lang, ground_truth_text)

@app.post("/video_translate/{target_lang}", response_model=VideoTranslationResponse, tags=["Video Translation"])
async def video_translate_endpoint(target_lang: str, video_file: UploadFile = File(...), ground_truth_text: str | None = Query(None)):
    tgt_lang_full = LANGUAGES.get(target_lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    return await video_translate(video_file, "eng_Latn", tgt_lang_full, target_lang, ground_truth_text)

@app.get("/demo_pib_translation", response_model=PIBTranslationResponse, tags=["Demo"])
async def demo_pib_translation(url: str = Query(...), lang: str = Query("hi"), include_audio: bool = Query(True)):
    tgt_lang_full = LANGUAGES.get(lang)
    if not tgt_lang_full:
        raise HTTPException(status_code=400, detail="Unsupported target language")
    
    start_time = time.time()
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        query_params['lang'] = ['1']
        new_query = urlencode(query_params, doseq=True)
        english_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, new_query, parsed_url.fragment))
        
        fetch_start = time.time()
        response = requests.get(english_url, timeout=10)
        response.raise_for_status()
        fetch_ms = round((time.time() - fetch_start) * 1000, 3)

        parse_start = time.time()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='innner-page-main-about-us-content-right-part')
        if not content_div: raise ValueError("Content div not found")
        
        paragraphs = [p.get_text(strip=True) for p in content_div.find_all('p')]
        original_text = '\n\n'.join(p for p in paragraphs if p and '***' not in p and 'MJPS' not in p)
        char_count = len(original_text)
        parse_ms = round((time.time() - parse_start) * 1000, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch or parse PIB article: {e}")

    translated_text, metrics, audio_base64 = translate_text(original_text, "eng_Latn", tgt_lang_full, lang, include_audio=include_audio)
    metrics.fetch_ms = fetch_ms
    metrics.parse_ms = parse_ms
    total_latency = round((time.time() - start_time) * 1000, 3)
    metrics.latency_ms = total_latency
    metrics.cost_rupees = round((total_latency / 60000) * 0.10, 5)

    return PIBTranslationResponse(
        source_language="en", target_language=lang,
        original_text=original_text, translated_text=translated_text,
        char_count=char_count, metrics=metrics, audio_base64=audio_base64
    )