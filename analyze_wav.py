import wave
import os

def analyze_wav(filepath):
    with wave.open(filepath, 'rb') as wav:
        print(f"\nAnalyzing: {os.path.basename(filepath)}")
        print(f"Number of channels: {wav.getnchannels()}")
        print(f"Sample width: {wav.getsampwidth()} bytes")
        print(f"Frame rate: {wav.getframerate()} Hz")
        print(f"Number of frames: {wav.getnframes()}")
        duration = wav.getnframes() / wav.getframerate()
        print(f"Duration: {duration:.3f} seconds")

# Analyze input and output files
input_file = "extensions/HRD III Drive channel modeling/DI.wav"
output_file = "extensions/HRD III Drive channel modeling/Drive 12 Master 1.wav"

analyze_wav(input_file)
analyze_wav(output_file) 