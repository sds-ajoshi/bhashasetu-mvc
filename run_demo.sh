#!/bin/bash

# Activate venv (assume created)
source venv/bin/activate

# Download models if not present (run Python snippets)
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', trust_remote_code=True); AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', trust_remote_code=True)"
python -c "import whisper; whisper.load_model('base')"

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &

# Wait for startup
sleep 5

# Sample tests (requires curl; adjust files/URLs)
echo "Test 1: Text Translation"
curl -X POST "http://127.0.0.1:8000/translate/text/hi" -H "Content-Type: application/json" -d '{"text": "India is a diverse country."}'

echo "Test 2: PIB Demo"
curl "http://127.0.0.1:8000/demo_pib_translation?url=https://pib.gov.in/PressReleasePage.aspx?PRID=1942492&lang=te"

# Add more tests as needed; kill uvicorn on exit
trap "kill $!" EXIT
wait