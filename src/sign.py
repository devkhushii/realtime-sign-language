import io
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime

# -------- Colors --------
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# -------- Audio Params --------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3  # Reduced for faster feedback
VOLUME_THRESHOLD = 200  # Adjust as needed

# -------- Load Whisper Model --------
model_size = "medium.en"  # Faster for real-time CPU use
model = WhisperModel(model_size, device="cpu")

# -------- List and Select Input Device --------
p = pyaudio.PyAudio()
print("Available input devices:")
device_info_map = {}

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        device_info_map[i] = info["name"]
        print(f"[{i}] {info['name']}")

device_index = int(input("\nEnter the device index of your microphone: "))
if device_index not in device_info_map:
    print("Invalid device index! Exiting...")
    exit(1)
print(f"Using device: {device_info_map[device_index]}\n")

# -------- Open Audio Stream --------
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

print("Model loaded. Speak something... (Ctrl+C to stop)\n")

accumulated_transcription = ""
last_transcription = ""

# -------- Normalize Audio --------
def normalize_audio(audio_array):
    max_val = np.max(np.abs(audio_array))
    if max_val == 0:
        return audio_array
    return (audio_array / max_val * 32767).astype(np.int16)

# -------- Record from Microphone --------
def record_to_buffer():
    frames = []
    volume_sum = 0
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_array = np.frombuffer(data, dtype=np.int16)
        volume_sum += np.abs(audio_array).mean()

    avg_volume = volume_sum / (RATE / CHUNK * RECORD_SECONDS)
    print(f"[DEBUG] Avg Volume: {avg_volume:.2f}")

    if avg_volume < VOLUME_THRESHOLD:
        return None

    combined_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    normalized_audio = normalize_audio(combined_audio)

    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(normalized_audio.tobytes())
    wf.close()
    buffer.seek(0)
    return buffer

# -------- Transcribe from Audio --------
def transcribe_from_buffer(audio_buffer):
    segments, _ = model.transcribe(audio_buffer, beam_size=10)
    return " ".join(segment.text for segment in segments).strip()

# -------- Main Loop --------
try:
    while True:
        buffer = record_to_buffer()
        if not buffer:
            print("Listening...")
            continue

        transcription = transcribe_from_buffer(buffer)

        if transcription and transcription.lower() != last_transcription.lower():
            print(NEON_GREEN + transcription + RESET_COLOR)
            accumulated_transcription += transcription + " "
            last_transcription = transcription
        else:
            print("speak now.....")

except KeyboardInterrupt:
    print("\nStopping... Saving log...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"log_{timestamp}.txt", "w") as f:
        f.write(accumulated_transcription.strip())

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("LOG: ", accumulated_transcription)







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
