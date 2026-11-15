"""Builds the final demo video: user's silent screen recording + end card + AI voice-over.

Run from this folder:  py -3 build_video.py
Needs: edge-tts, imageio-ffmpeg, pillow.  Output: final_andrew.mp4, final_ava.mp4, subtitles.srt
"""
import os
import re
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1914, 892
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ---------------------------------------------------------------- timeline
SPEED = 2.5                       # typing part (original 50-82 s) is sped up
CUT_A, CUT_B, CUT_C = 50.0, 82.0, 92.0
V2_LEN = (CUT_B - CUT_A) / SPEED  # 12.8 s
V3_LEN = CUT_C - CUT_B            # 10 s
CARD_LEN = 30.0
CARD_START = CUT_A + V2_LEN + V3_LEN
TOTAL = CARD_START + CARD_LEN

# (name, start second on the NEW timeline, latest end second, text)
SEGMENTS = [
    ("s1", 0.8, 12.4, "Many parents and grandparents in Kazakhstan can't read English, but they get medicine leaflets and school letters. Senim translates English into Kazakh, and warns you when the translation shouldn't be trusted."),
    ("s2", 12.8, 26.5, "First, a dosage instruction. The numbers and units are kept, and the back-translation matches, so the label says: likely OK."),
    ("s3", 27.5, 36.0, "Another warning, this time about children under twelve: still fine."),
    ("s4", 38.0, 44.8, "A school meeting: the day, date, time and room are all preserved."),
    ("s5", 45.5, 53.0, "Here the check disagrees: signed permission slip came back as written permit. The meaning is close, but Senim warns anyway."),
    ("s6", 53.5, 62.5, "Now my own sentence, with typos, like a real message."),
    ("s7", 67.0, 72.6, "Fluent Kazakh again, but the back-translation no longer matches what I typed, so Senim warns."),
    ("s8", CARD_START + 0.7, TOTAL - 0.4, "Behind this is the free NLLB-200 model, tested on 150 professional FLORES translations: Kazakh scores 52 chrF, leaving the English unchanged scores 1.4. I rated 90 translations myself: 44 were partly or fully wrong. Senim warns about 6 of 8 serious errors; always trusting catches none. Limits: one rater and a small test set. Next: more Kazakh-speaking raters. A prototype, not medical advice."),
]

VOICES = {"andrew": "en-US-AndrewNeural", "ava": "en-US-AvaNeural"}


def duration(path):
    r = subprocess.run([FF, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def tts(name, text, voice, window):
    """Render one segment; speed the voice up until it fits its window."""
    with open(f"seg_{name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    for rate in (0, 5, 10, 15, 20):
        out = f"seg_{voice}_{name}.mp3"
        words = len(text.split())
        for attempt in range(8):  # the online voice service sometimes returns nothing or cuts the audio: retry
            if os.path.exists(out):
                os.remove(out)
            subprocess.run(["py", "-3", "-m", "edge_tts", "--voice", VOICES[voice], f"--rate={rate:+d}%",
                            "--file", f"seg_{name}.txt", "--write-media", out], capture_output=True)
            if os.path.exists(out) and os.path.getsize(out) > 5000 and duration(out) >= 0.2 * words:
                break
        else:
            raise RuntimeError(f"text-to-speech failed for {name} ({voice})")
        d = duration(out)
        if d <= window:
            return out, d, rate
    return out, d, rate


# ---------------------------------------------------------------- end card
def make_card(path):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fonts = r"C:\Windows\Fonts"
    f_title = ImageFont.truetype(os.path.join(fonts, "arialbd.ttf"), 78)
    f_sub = ImageFont.truetype(os.path.join(fonts, "arial.ttf"), 38)
    f_h = ImageFont.truetype(os.path.join(fonts, "arialbd.ttf"), 40)
    f_b = ImageFont.truetype(os.path.join(fonts, "arial.ttf"), 38)
    f_s = ImageFont.truetype(os.path.join(fonts, "arial.ttf"), 30)
    orange, dark, grey = (234, 88, 12), (31, 41, 55), (107, 114, 128)
    d.text((110, 70), "Senim · Сенім", font=f_title, fill=orange)
    d.text((110, 170), "English → Kazakh translation that warns when not to trust it", font=f_sub, fill=dark)
    y = 270
    blocks = [
        ("Translation quality: FLORES+ test, 150 sentences (chrF)",
         "Leave the English as is: 1.4     NLLB-200 → Kazakh 52.1, Russian 56.0, French 69.2"),
        ("The author's ratings of 90 Kazakh translations",
         "44 were partly or fully wrong"),
        ("Trust check, held-out half of the ratings",
         "Warns about 6 of 8 serious errors   (always trust: 0)"),
        ("Best of 8 candidate translations",
         "+2.6 chrF on held-out test sentences"),
    ]
    for head, body in blocks:
        d.rectangle([110, y + 6, 122, y + 100], fill=orange)
        d.text((145, y), head, font=f_h, fill=dark)
        d.text((145, y + 54), body, font=f_b, fill=dark)
        y += 130
    d.text((110, H - 110), "github.com/arturbaimuratov1-sys/senim", font=f_b, fill=dark)
    d.text((110, H - 62), "Prototype, not medical advice   ·   Voice-over: AI-generated (Microsoft Edge neural voice)",
           font=f_s, fill=grey)
    img.save(path)


# ---------------------------------------------------------------- video
def build_video_only(path):
    cover = ("drawbox=x=1536:y=745:w=378:h=147:color=white:t=fill:"
             "enable='between(t,2.5,9.5)+between(t,68,74.5)'")
    fc = (
        f"[0:v]{cover},split=3[a][b][c];"
        f"[a]trim=0:{CUT_A},setpts=PTS-STARTPTS,fps=30[v1];"
        f"[b]trim={CUT_A}:{CUT_B},setpts=(PTS-STARTPTS)/{SPEED},fps=30[v2];"
        f"[c]trim={CUT_B}:{CUT_C},setpts=PTS-STARTPTS,fps=30[v3];"
        f"[1:v]fps=30,format=yuv420p,setsar=1[v4];"
        "[v1][v2][v3][v4]concat=n=4:v=1:a=0,format=yuv420p[outv]"
    )
    subprocess.run([FF, "-y", "-i", "screen.mp4", "-loop", "1", "-t", str(CARD_LEN), "-framerate", "30",
                    "-i", "card.png", "-filter_complex", fc, "-map", "[outv]", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "23", "-r", "30", path], capture_output=True, text=True, check=True)


def mux(video, voice, parts, out):
    """parts: list of (mp3, start_seconds)."""
    inputs, filters = [], []
    for i, (mp3, start) in enumerate(parts):
        inputs += ["-i", mp3]
        filters.append(f"[{i + 1}:a]adelay={int(start * 1000)}|{int(start * 1000)},volume=1.0[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(parts))) + f"amix=inputs={len(parts)}:normalize=0,apad=whole_dur={TOTAL}[aout]"
    fc = ";".join(filters + [mix])
    subprocess.run([FF, "-y", "-i", video, *inputs, "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(TOTAL), out],
                   capture_output=True, text=True, check=True)


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


if __name__ == "__main__":
    print(f"timeline: typing part {V2_LEN:.1f}s, card starts {CARD_START:.1f}s, total {TOTAL:.1f}s")
    make_card("card.png")
    build_video_only("video_only.mp4")
    print("video-only:", round(duration("video_only.mp4"), 1), "s")
    for voice in VOICES:
        parts, report = [], []
        for name, start, end, text in SEGMENTS:
            mp3, d, rate = tts(name, text, voice, end - start)
            ok = d <= end - start
            report.append(f"{voice} {name}: starts {start:.1f}s, speech {d:.1f}s, window {end - start:.1f}s, rate {rate:+d}% {'OK' if ok else 'TOO LONG'}")
            parts.append((mp3, start))
            if voice == "andrew":
                pass
        print("\n".join(report))
        mux("video_only.mp4", voice, parts, f"final_{voice}.mp4")
        print(f"final_{voice}.mp4:", round(duration(f"final_{voice}.mp4"), 1), "s")
    # subtitles (English) with Andrew's timing
    lines = []
    for i, (name, start, end, text) in enumerate(SEGMENTS, 1):
        d = duration(f"seg_en-US-AndrewNeural_{name}.mp3") if os.path.exists(f"seg_en-US-AndrewNeural_{name}.mp3") else end - start
        lines.append(f"{i}\n{srt_time(start)} --> {srt_time(start + min(d, end - start))}\n{text}\n")
    open("subtitles.srt", "w", encoding="utf-8").write("\n".join(lines))
    print("subtitles.srt written")
