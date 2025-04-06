import queue
import os
import numpy as np
import pyaudio
import wave
from faster_whisper import WhisperModel

NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Load Whisper model
model_size = "medium.en"
model = WhisperModel(model_size, device="cpu")

# Audio stream parameters
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio_queue = queue.Queue()
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

accumulated_transcription = ""

def record_chunk(p, stream, filename, duration=2):  # Increase duration to 2s
    frames = []
    for _ in range(0, int(RATE / CHUNK_SIZE * duration)):  
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)  # Prevent buffer overflow
        frames.append(data)

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

def transcribe_chunk(model, filename):
    segments, _ = model.transcribe(filename, word_timestamps=True, beam_size=5)
    return " ".join(segment.text for segment in segments)

try:
    while True:
        print("speak now.....")
        chunk_file = "temp_chunk.wav"
        record_chunk(p, stream, chunk_file)
        transcription = transcribe_chunk(model, chunk_file)

        if transcription.strip():  # Ignore empty transcriptions
            print(NEON_GREEN + transcription + RESET_COLOR)
            accumulated_transcription += transcription + " "
        
        os.remove(chunk_file)

except KeyboardInterrupt:
    print("Stopping....")
    with open("log.txt", "w") as log_file:
        log_file.write(accumulated_transcription)
finally:
    print("LOG: " + accumulated_transcription)
    stream.stop_stream()
    stream.close()
    p.terminate()




# Initialize Pygame for avatar display
# pygame.init()
# screen = pygame.display.set_mode((800, 600))
# pygame.display.set_caption("Speech to Sign Language Avatar")

# # Load Sign Language Images (Replace with actual images)
# sign_images = {
#     "hello": pygame.image.load("sign_hello.png"),
#     "yes": pygame.image.load("sign_yes.png"),
#     "no": pygame.image.load("sign_no.png"),
#     "thank you": pygame.image.load("sign_thank_you.png"),
#     "default": pygame.image.load("sign_default.png"),  # Default avatar image
# }

# # Function to display a sign
# def show_sign(word):
#     screen.fill((255, 255, 255))  # Clear screen
#     img = sign_images.get(word.lower(), sign_images["default"])
#     screen.blit(img, (250, 150))
#     pygame.display.flip()

# print("🎙️ Listening... Speak now!")

# # Start streaming
# stream.start_stream()

# # Buffer for audio storage
# audio_buffer = np.array([], dtype=np.float32)

# try:
#     while True:
#         # Process Pygame events
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 raise KeyboardInterrupt

#         # Collect audio data from queue
#         if not audio_queue.empty():
#             audio_data = audio_queue.get()
#             audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#             audio_buffer = np.concatenate((audio_buffer, audio_np))  # Append new data

#             # Process in small chunks (~1 second of audio)
#             if len(audio_buffer) > RATE:  
#                 segments, _ = model.transcribe(audio_buffer, beam_size=5)
#                 audio_buffer = np.array([], dtype=np.float32)  # Reset buffer after processing
                
#                 for segment in segments:
#                     word = segment.text.lower()
#                     print(f"Detected: {word}")  # Print detected word
# #                     show_sign(word)  # Display sign avatar

# except KeyboardInterrupt:
#     print("\n🔴 Stopping...")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     pygame.quit()
