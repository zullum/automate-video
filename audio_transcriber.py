import whisper
import os

def transcribe_to_srt(audio_path, model_type="base", output_srt="output.srt"):
    # Load the Whisper model
    model = whisper.load_model(model_type)

    # Transcribe the audio file
    result = model.transcribe(audio_path)

    # Prepare SRT file content
    segments = result['segments']
    srt_content = ""
    for idx, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()

        # Convert timestamps to SRT format
        start_srt = format_timestamp(start)
        end_srt = format_timestamp(end)

        srt_content += f"{idx + 1}\n{start_srt} --> {end_srt}\n{text}\n\n"

    # Save to SRT file
    with open(output_srt, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    print(f"SRT file saved to {output_srt}")

def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hrs = int(seconds) // 3600
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

if __name__ == "__main__":
    # Input audio file path
    audio_path = "story.mp3"  # Replace with your audio file

    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found!")
    else:
        # Transcribe and generate SRT file
        transcribe_to_srt(audio_path)
