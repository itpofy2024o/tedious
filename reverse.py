from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from moviepy.video.fx.all import time_mirror, speedx
import os

ip = "i.mp4"
op = "o.mp4"
ps = 1.5
vdc = -5

def process(ip,op,sp,vdb):
	clip=VideoFileClip(ip)
	rc = time_mirror(clip)
	sc = speedx(rc,factor=sp)
	tmpaudopath = "t.wav"
	sc.audio.write_audiofile(tmpaudopath)
	audio = AudioSegment.from_file(tmpaudopath)
	awdj = audio+vdb
	awdj.export(tmpaudopath,format="wav")
	fa=AudioFileClip(tmpaudopath)
	fc=sc.set_audio(fa)
	fc.write_videofile(op,codec="libx264",audio_codec="aac")
	os.remove(tmpaudopath)

if __name__ == "__main__":
	process(ip,op,ps,vdc)

