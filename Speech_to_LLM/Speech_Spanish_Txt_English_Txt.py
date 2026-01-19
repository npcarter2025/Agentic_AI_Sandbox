#!/usr/bin/env python3
"""
Simple Real-time Speech Translation
Combines Vosk ASR with Transformers translation
"""

import queue
import sys
import sounddevice as sd
import time
import json

from vosk import Model, KaldiRecognizer
from transformers import MarianMTModel, MarianTokenizer

# Global variables
audio_queue = queue.Queue()
translation_model = None
translation_tokenizer = None

def audio_callback(indata, frames, time, status):
    """Audio callback for microphone input"""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

def load_translation_model():
    """Load the translation model once"""
    global translation_model, translation_tokenizer
    
    print("Loading translation model...")
    model_name = "Helsinki-NLP/opus-mt-es-en"
    translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    print("✅ Translation model loaded!")

def translate_text(text):
    """Translate Spanish text to English"""
    global translation_model, translation_tokenizer
    
    try:
        inputs = translation_tokenizer(text, return_tensors="pt", padding=True)
        outputs = translation_model.generate(**inputs, max_length=50, num_beams=1, do_sample=False)
        translated = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        return f"Translation error: {e}"

def main():
    print("🎤 Simple Speech Translation")
    print("=" * 40)
    
    # Load translation model
    load_translation_model()
    
    # Load Vosk model
    print("Loading speech recognition model...")
    model = Model(lang="es")
    print("✅ Speech model loaded!")
    
    # Setup audio
    samplerate = 16000  # Vosk works best with 16kHz
    
    print("\n" + "=" * 50)
    print("🎯 READY! Speak in Spanish...")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, 
                              dtype="int16", channels=1, callback=audio_callback):
            
            rec = KaldiRecognizer(model, samplerate)
            last_translation_time = 0
            
            while True:
                data = audio_queue.get()
                
                if rec.AcceptWaveform(data):
                    # Complete sentence
                    result = rec.Result()
                    if result:
                        result_dict = json.loads(result)
                        if 'text' in result_dict and result_dict['text'].strip():
                            spanish_text = result_dict['text'].strip()
                            print(f"\n🗣️  Spanish: {spanish_text}")
                            
                            # Translate
                            english_text = translate_text(spanish_text)
                            print(f"🇬🇧 English: {english_text}")
                            print("-" * 40)
                            
                            last_translation_time = time.time()
                else:
                    # Partial result
                    partial_result = rec.PartialResult()
                    if partial_result:
                        try:
                            partial_dict = json.loads(partial_result)
                            if 'partial' in partial_dict and partial_dict['partial'].strip():
                                current_text = partial_dict['partial'].strip()
                                print(f"\r🎤 Listening: {current_text}", end="", flush=True)
                        except:
                            pass
                
                # Translate after 3 seconds of silence
                if time.time() - last_translation_time > 3.0:
                    partial_result = rec.PartialResult()
                    if partial_result:
                        try:
                            partial_dict = json.loads(partial_result)
                            if 'partial' in partial_dict and partial_dict['partial'].strip():
                                spanish_text = partial_dict['partial'].strip()
                                print(f"\n🗣️  Spanish: {spanish_text}")
                                english_text = translate_text(spanish_text)
                                print(f"🇬🇧 English: {english_text}")
                                print("-" * 40)
                                last_translation_time = time.time()
                        except:
                            pass
                            
    except KeyboardInterrupt:
        print("\n\n👋 Translation stopped!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
