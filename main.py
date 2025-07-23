from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import time
import torch
from torch.cuda.amp import autocast
import torch.quantization
import functools
from concurrent.futures import ThreadPoolExecutor
import io
import base64
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import ffmpeg
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
import nltk
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import re
import logging
import os
import subprocess
from pydub import AudioSegment
import whisper

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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
    "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "bn": "ben_Beng",
    "ml": "mal_Mlym", "gu": "guj_Gujr", "mr": "mar_Deva", "pa": "pan_Guru",
    "kn": "kan_Knda", "ur": "urd_Arab", "or": "ori_Orya", "as": "asm_Beng"
}

# --- Model and Toolkit Initialization ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
logger.info(f"Loading IndicTrans2 Model: {MODEL_NAME}")
translator_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
translator_model.eval()

# Apply dynamic quantization if on CPU
if DEVICE == "cpu":
    translator_model = torch.quantization.quantize_dynamic(
        translator_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    logger.info("Applied dynamic quantization to the model for CPU.")

logger.info("IndicTrans2 loaded and quantized successfully.")

ip = IndicProcessor(inference=True)
logger.info("IndicProcessor loaded successfully.")

logger.info("Loading Whisper Model (base)...")
whisper_model = whisper.load_model("base", device=DEVICE)
logger.info("Whisper model loaded successfully.")

# --- Translation Cache ---
translation_cache = {}

# --- Preprocessing Cache ---
@functools.lru_cache(maxsize=1000)
def preprocess_cache(text: str, src_lang: str, tgt_lang: str) -> str:
    return ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)[0]

# --- Helper Functions ---
def calculate_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference)
    hyp_tokens = nltk.word_tokenize(hypothesis)
    return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))

def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return edit_distance(ref_words, hyp_words) / len(ref_words) if ref_words else 0.0

def translate_batch(batch_chunks, src_lang, tgt_lang_full):
    try:
        batch = [preprocess_cache(chunk, src_lang, tgt_lang_full) for chunk in batch_chunks]
        inputs = translator_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            with autocast(enabled=DEVICE == "cuda"):
                generated_tokens = translator_model.generate(
                    **inputs, use_cache=True, min_length=0, max_length=512, num_beams=1, num_return_sequences=1
                )
        decoded_tokens = translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return ip.postprocess_batch(decoded_tokens, lang=tgt_lang_full)
    except Exception as e:
        logger.error(f"Error in translate_batch: {e}")
        return [""] * len(batch_chunks)

def translate_chunks(chunks: list[str], src_lang: str, tgt_lang_full: str, batch_size: int = 32, max_workers: int = 4):
    if not chunks: return []
    non_empty_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
    if not non_empty_chunks: return [""] * len(chunks)
    
    logger.info(f"Translating {len(non_empty_chunks)} non-empty chunks in parallel with {max_workers} workers...")
    translated_chunks = []
    to_translate = []
    cache_hits = 0
    
    for chunk in non_empty_chunks:
        cache_key = (chunk, src_lang, tgt_lang_full)
        if cache_key in translation_cache:
            translated_chunks.append(translation_cache[cache_key])
            cache_hits += 1
        else:
            to_translate.append(chunk)
    
    logger.info(f"Cache hits: {cache_hits}, Translating {len(to_translate)} chunks...")
    
    if to_translate:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(translate_batch, to_translate[i:i + batch_size], src_lang, tgt_lang_full)
                for i in range(0, len(to_translate), batch_size)
            ]
            for future in futures:
                batch_translations = future.result()
                translated_chunks.extend(batch_translations)
                # Update cache
                for chunk, translation in zip(to_translate[len(translated_chunks) - len(batch_translations):], batch_translations):
                    cache_key = (chunk, src_lang, tgt_lang_full)
                    translation_cache[cache_key] = translation
    
    translated_map = {original: translated for original, translated in zip(non_empty_chunks, translated_chunks)}
    return [translated_map.get(chunk, "") if chunk and chunk.strip() else "" for chunk in chunks]

# --- **OPTIMIZED** HTML TRANSLATION HELPER ---
def translate_html_content(soup_elements: list, src_lang: str, tgt_lang_full: str, min_length: int = 5):
    """Translates a list of BeautifulSoup elements efficiently by modifying them in-place."""
    
    # Step 1: Collect all text nodes that need translation from all elements
    nodes_to_translate = []
    for elem in soup_elements:
        # Find all text strings within the element that are not just whitespace
        for node in elem.find_all(string=True):
            if isinstance(node, NavigableString):
                text = node.strip()
                if text and len(text) >= min_length and not re.match(r'^[\W\d]+$', text):
                    nodes_to_translate.append(node)

    # Step 2: Extract the text from these nodes
    original_texts = [node.strip() for node in nodes_to_translate]

    # Step 3: Translate all texts in batches
    translated_texts = translate_chunks(original_texts, src_lang, tgt_lang_full)

    # Step 4: Replace the content of the original nodes with the translated text
    for node, translated_text in zip(nodes_to_translate, translated_texts):
        if translated_text:
            node.replace_with(NavigableString(translated_text))

    # Step 5: Reconstruct the final HTML from the now-modified elements
    return '\n'.join(str(elem) for elem in soup_elements)

def generate_audio_from_narrative(narrative_text: str, tgt_lang_short: str, max_workers: int = 4):
    audio_base64 = None
    if not narrative_text or not narrative_text.strip():
        return None
    logger.info("Generating audio for translated narrative...")
    try:
        sentences = nltk.sent_tokenize(narrative_text)
        combined_audio = AudioSegment.empty()
        
        def generate_tts_segment(sentence):
            if not sentence.strip(): return None
            tts = gTTS(text=sentence, lang=tgt_lang_short)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return AudioSegment.from_file(fp, format="mp3")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            segments = list(executor.map(generate_tts_segment, sentences))
            for segment in segments:
                if segment:
                    combined_audio += segment
        
        final_buffer = io.BytesIO()
        combined_audio.export(final_buffer, format="mp3")
        final_buffer.seek(0)
        audio_bytes = final_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("Audio generation successful.")
    except Exception as e:
        logger.warning(f"gTTS error during chunk processing: {e}")
    return audio_base64

def translate_text(text: str, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None, include_audio: bool = True):
    if not text:
        return "", Metrics(latency_ms=0.0, cost_rupees=0.0, time_complexity="O(1)", space_complexity="O(1)", fetch_ms=0.0, parse_ms=0.0, translate_ms=0.0), None

    start_time = time.time()
    translated_text = ""
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = [' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
        translated_chunks = translate_chunks(chunks, src_lang, tgt_lang_full)
        translated_text = ' '.join(translated_chunks)
    except Exception as e:
        logger.error(f"Plain text translation error: {e}")
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
    
    audio_base64 = generate_audio_from_narrative(translated_text, tgt_lang_short) if include_audio else None
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
        segment_texts = [seg['text'].strip() for seg in segments if seg['text'].strip()]
        translated_segments = translate_chunks(segment_texts, src_lang, tgt_lang_full)
        
        trans_idx = 0
        for i, segment in enumerate(segments, 1):
            start, end, text = segment["start"], segment["end"], segment["text"].strip()
            if text:
                translated_segment = translated_segments[trans_idx]
                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start-int(start))*1000):03d}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end-int(end))*1000):03d}"
                srt_content += f"{i}\n{start_time} --> {end_time}\n{translated_segment}\n\n"
                trans_idx += 1

        _, metrics, audio_base64 = translate_text(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio=True)
        metrics.wer_score = wer_score

        translation = TranslationResponse(
            source_language="en", target_language=tgt_lang_short,
            source_text=transcribed_text, translated_text=' '.join(translated_segments),
            char_count=len(transcribed_text), metrics=metrics, audio_base64=audio_base64
        )
        return SubtitleResponse(srt_content=srt_content, translation=translation)
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

async def video_translate(video_file: UploadFile, src_lang: str, tgt_lang_full: str, tgt_lang_short: str, ground_truth: str | None = None):
    base_name = os.path.splitext(video_file.filename)[0]
    paths = {
        "video": f"./temp_{base_name}.mp4", "audio": f"./temp_{base_name}.mp3",
        "tts": f"./temp_tts_{base_name}.mp3", "output": f"./temp_output_{base_name}.mp4"
    }
    try:
        with open(paths["video"], "wb") as buffer: buffer.write(await video_file.read())

        logger.info("Extracting audio from video...")
        ffmpeg.input(paths["video"]).output(paths["audio"], acodec='libmp3lame').run(overwrite_output=True, quiet=True)

        logger.info("Transcribing audio...")
        result = whisper_model.transcribe(paths["audio"], fp16=torch.cuda.is_available(), language="en")
        transcribed_text = result["text"]
        wer_score = calculate_wer(ground_truth, transcribed_text) if ground_truth else None

        logger.info("Translating text...")
        translated_text, metrics, audio_base64 = translate_text(transcribed_text, src_lang, tgt_lang_full, tgt_lang_short, ground_truth, include_audio=True)
        metrics.wer_score = wer_score

        if not audio_base64: raise ValueError("TTS generation failed.")
        
        with open(paths["tts"], "wb") as f: f.write(base64.b64decode(audio_base64))

        logger.info("Overlaying new audio onto video...")
        input_video = ffmpeg.input(paths["video"])
        input_audio = ffmpeg.input(paths["tts"])
        ffmpeg.output(input_video.video, input_audio.audio, paths["output"], vcodec='copy', acodec='aac', shortest=None).run(overwrite_output=True, quiet=True)

        with open(paths["output"], "rb") as f: video_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        translation_response = TranslationResponse(
            source_language="en", target_language=tgt_lang_short,
            source_text=transcribed_text, translated_text=translated_text,
            char_count=len(transcribed_text), metrics=metrics, audio_base64=audio_base64
        )
        return VideoTranslationResponse(transcribed_text=transcribed_text, translation=translation_response, video_base64=video_base64)
    finally:
        for path in paths.values():
            if os.path.exists(path): os.remove(path)

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
        
        logger.info(f"Fetching PIB URL: {english_url}")
        fetch_start = time.time()
        response = requests.get(english_url, timeout=15)
        response.raise_for_status()
        fetch_ms = round((time.time() - fetch_start) * 1000, 3)

        parse_start = time.time()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='innner-page-main-about-us-content-right-part')
        if not content_div: raise ValueError("Content div not found")
        
        original_text_html = str(content_div)
        char_count = len(content_div.get_text())
        parse_ms = round((time.time() - parse_start) * 1000, 3)
        logger.info(f"Parsed {char_count} chars in {parse_ms}ms")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch or parse PIB article: {e}")

    translation_start_time = time.time()
    # Use a copy for translation
    translation_soup = BeautifulSoup(str(content_div), 'html.parser')
    content_elements = [child for child in translation_soup.children if not isinstance(child, NavigableString)]
    translated_text_html = translate_html_content(content_elements, "eng_Latn", tgt_lang_full, min_length=5)
    translate_ms = round((time.time() - translation_start_time) * 1000, 3)

    # Extract narrative from the TRANSLATED HTML for audio generation
    translated_soup_for_audio = BeautifulSoup(translated_text_html, 'html.parser')
    translated_narrative_text = ' '.join([p.get_text(strip=True) for p in translated_soup_for_audio.find_all('p')])
    
    audio_base64 = generate_audio_from_narrative(translated_narrative_text, lang) if include_audio else None
    
    total_latency = round((time.time() - start_time) * 1000, 3)
    cost = round((total_latency / 60000) * 0.10, 5)

    metrics = Metrics(
        latency_ms=total_latency, cost_rupees=cost,
        time_complexity="O(n^2 * d)", space_complexity="O(n * d)",
        fetch_ms=fetch_ms, parse_ms=parse_ms, translate_ms=translate_ms
    )

    return PIBTranslationResponse(
        source_language="en", target_language=lang,
        original_text=original_text_html, translated_text=translated_text_html,
        char_count=char_count, metrics=metrics, audio_base64=audio_base64
    )