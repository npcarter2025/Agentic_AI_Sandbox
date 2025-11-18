#!/usr/bin/env python3
"""
Spacebar-Controlled Speech to Text
Press spacebar to record, press again to process
Uses Vosk ASR for English speech recognition
"""

import sys
import sounddevice as sd
import time
import json
import threading
from collections import deque

from vosk import Model, KaldiRecognizer
from pynput import keyboard

# Global variables
is_recording = False
recorded_audio = deque()
recording_lock = threading.Lock()
should_exit = False

def audio_callback(indata, frames, time, status):
    """Audio callback for microphone input"""
    if status:
        print(status, file=sys.stderr)
    
    with recording_lock:
        if is_recording:
            recorded_audio.append(bytes(indata))

def process_recorded_audio(model, samplerate):
    """Process the recorded audio chunks"""
    global recorded_audio
    
    if not recorded_audio:
        return
    
    print("\n🔄 Processing audio...")
    
    # Create a new recognizer for this recording
    rec = KaldiRecognizer(model, samplerate)
    
    # Process all recorded audio chunks
    with recording_lock:
        audio_chunks = list(recorded_audio)
        recorded_audio.clear()
    
    for chunk in audio_chunks:
        rec.AcceptWaveform(chunk)
    
    # Get final result
    result = rec.FinalResult()
    if result:
        try:
            result_dict = json.loads(result)
            if 'text' in result_dict and result_dict['text'].strip():
                english_text = result_dict['text'].strip()
                print(f"\n🗣️  English: {english_text}")
                print("-" * 40)
            else:
                print("⚠️  No speech detected in recording")
        except Exception as e:
            print(f"❌ Error processing result: {e}")

def on_press(key):
    """Handle key press events - toggle recording on spacebar"""
    global is_recording, should_exit
    
    try:
        if key == keyboard.Key.space:
            with recording_lock:
                if not is_recording:
                    # Start recording
                    is_recording = True
                    recorded_audio.clear()
                    print("\n🎤 RECORDING... (Press spacebar again to stop)")
                else:
                    # Stop recording
                    is_recording = False
                    print("⏹️  Recording stopped")
    except AttributeError:
        pass
    
    if key == keyboard.Key.esc:
        # Stop the program
        should_exit = True
        return False

def on_release(key):
    """Handle key release events"""
    # We don't need to do anything on release for toggle mode
    pass

def main():
    print("🎤 Spacebar-Controlled Speech to Text")
    print("=" * 50)
    
    # Load Vosk model
    print("Loading speech recognition model...")
    model = Model(lang="en-us")
    print("✅ Speech model loaded!")
    
    # Setup audio
    samplerate = 16000  # Vosk works best with 16kHz
    
    print("\n" + "=" * 50)
    print("🎯 READY!")
    print("Press SPACEBAR to start recording")
    print("Press SPACEBAR again to stop and transcribe")
    print("Press ESC to exit")
    print("=" * 50)
    
    try:
        # Start keyboard listener in background
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        listener.start()
        
        # Start audio stream (always listening, but only recording when spacebar is pressed)
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, 
                              dtype="int16", channels=1, callback=audio_callback):
            
            last_recording_state = False
            
            while not should_exit:
                # Check if recording just stopped
                with recording_lock:
                    current_recording_state = is_recording
                
                # If we just stopped recording, process the audio
                if last_recording_state and not current_recording_state:
                    # Small delay to ensure all audio chunks are captured
                    time.sleep(0.2)
                    process_recorded_audio(model, samplerate)
                
                last_recording_state = current_recording_state
                
                # Small delay to avoid busy waiting
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\n👋 Transcription stopped!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

