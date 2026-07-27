import whisper
import os

model = whisper.load_model("base")

def transcribe_video(video_path, output_folder="transcripts"):
    os.makedirs(output_folder, exist_ok=True)

    result = model.transcribe(video_path)

    transcript = result["text"]

    # Always overwrite the same transcript
    transcript_path = os.path.join(output_folder, "meeting_transcript.txt")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return transcript, transcript_path
