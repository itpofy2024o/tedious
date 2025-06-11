from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

# ------- CONFIG -------
AUDIO_FILE = "../Downloads/ahh.mp3"
VIDEO_FILE = "../Downloads/ahh.mp4"
OUTPUT_FILE = "h.mp4"
START_MS = 0
DURATION_MS = 5 * 60 * 1000 + 40 * 1000  # 3 min 45 sec = 225000 ms
# ----------------------

# Step 1: Cut the audio
audio = AudioSegment.from_mp3(AUDIO_FILE)
clip = audio[START_MS:START_MS + DURATION_MS]
clip.export("clipped_audio.mp3", format="mp3")

# Step 2: Merge audio and video
video = VideoFileClip(VIDEO_FILE).subclip(0, 340)  # 225 seconds = 3m45s
new_audio = AudioFileClip("clipped_audio.mp3").set_duration(video.duration)
final = video.set_audio(new_audio)

final.write_videofile(OUTPUT_FILE, codec='libx264', audio_codec='aac')

