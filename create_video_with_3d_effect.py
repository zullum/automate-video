import subprocess
import os
import logging
import json
from pathlib import Path
from datetime import datetime
import re
import time
import numpy as np
import argparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file using ffprobe."""
    logger.info(f"Getting duration of audio file: {audio_file}")
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
    hours, minutes, seconds = time_str.replace(',', '.').split(':')
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

def create_ass_subtitles(srt_file: str, output_file: str):
    """Convert SRT to ASS format with enhanced styling and word-level animations."""
    logger.info(f"Converting {srt_file} to ASS format with enhanced styling")
    
    # ASS header with styles
    ass_header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1408
PlayResY: 768
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,58,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,20,1
Style: Active,Montserrat,58,&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,110,110,0,0,1,4,0,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''

    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    srt_blocks = re.split(r'\n\n+', srt_content.strip())
    ass_events = []

    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            time_line = lines[1]
            start_time, end_time = time_line.split(' --> ')
            
            text = ' '.join(lines[2:])
            words = text.upper().split()
            
            total_duration = parse_srt_time(end_time) - parse_srt_time(start_time)
            word_duration = total_duration / len(words)
            
            for i, word in enumerate(words):
                word_start = parse_srt_time(start_time) + (i * word_duration)
                word_end = word_start + word_duration
                
                start_str = f"{int(word_start//3600):d}:{int((word_start%3600)//60):02d}:{word_start%60:05.2f}"
                end_str = f"{int(word_end//3600):d}:{int((word_end%3600)//60):02d}:{word_end%60:05.2f}"
                
                other_words = words.copy()
                other_words[i] = f"{{\\rStyle(Active)}}{word}{{\\rStyle(Default)}}"
                styled_line = ' '.join(other_words)
                
                ass_events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{styled_line}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        for event in ass_events:
            f.write(event + '\n')
    
    logger.info(f"Created ASS subtitle file: {output_file}")
    return output_file

def create_video_from_videos(
    video_folder: str,
    audio_file: str = "story.mp3",
    output_filename: str = None,
    output_width: int = 1408,
    output_height: int = 768,
    fps: int = 60,
    subtitle_file: str = "output.srt",
    speed_factor: float = 1.0  # New parameter: 1.0 is normal speed, 0.5 is half speed, 2.0 is double speed
):
    """
    Create a video from MP4 files with subtitles, controlling the playback speed of each video segment.
    
    Args:
        speed_factor: Controls playback speed of input videos.
                     1.0 = normal speed
                     < 1.0 = slower (e.g., 0.5 for half speed)
                     > 1.0 = faster (e.g., 2.0 for double speed)
    """
    start_time = time.time()
    logger.info(f"Starting video creation process from video folder: {video_folder}")
    logger.debug(f"Output dimensions: {output_width}x{output_height}, FPS: {fps}")
    logger.info(f"Using subtitle file: {subtitle_file}")
    logger.info(f"Video speed factor: {speed_factor}x")
    
    # Convert SRT to ASS with enhanced styling
    ass_subtitle_file = "temp_subtitles.ass"
    create_ass_subtitles(subtitle_file, ass_subtitle_file)
    logger.info("Created enhanced ASS subtitles with word-level animations")
    
    # Get audio duration
    audio_duration = get_audio_duration(audio_file)
    logger.info(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Gather MP4 files
    video_files = sorted(
        [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
    )
    if not video_files:
        logger.error("No MP4 files found in the specified folder.")
        return
    
    num_videos = len(video_files)
    segment_duration = audio_duration / num_videos
    logger.info(f"Found {num_videos} videos; each will display for ~{segment_duration:.2f} seconds")
    
    # Build output filename if not provided
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_video_{timestamp}.mp4"
    
    # Prepare FFmpeg inputs and filter chains
    inputs = []
    filter_chains = []
    
    for i, video in enumerate(video_files):
        video_path = os.path.join(video_folder, video)
        inputs.extend(['-i', video_path])
        
        # Get input video duration
        probe_cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        input_duration = float(video_info['format']['duration'])
        
        # Calculate loop count needed to fill segment duration
        effective_duration = input_duration / speed_factor
        loop_count = max(1, int(np.ceil(segment_duration / effective_duration)))
        logger.debug(f"Video {i+1}: Input duration={input_duration:.2f}s, "
                    f"Effective duration={effective_duration:.2f}s, "
                    f"Loops needed={loop_count}")
        
        chain = (
            f'[{i}:v]'
            # Apply speed effect
            f'setpts={1/speed_factor}*PTS,'
            # Force input to desired FPS
            f'fps={fps},'
            # Loop video to fill segment duration
            f'loop=loop={loop_count}:size={int(segment_duration * fps)},'
            # Scale to desired dimensions
            f'scale={output_width}:{output_height}:flags=lanczos,'
            # Add fade effects
            f'fade=t=in:st=0:d=1:alpha=1,'
            f'fade=t=out:st={max(segment_duration - 1, 0)}:d=1:alpha=1,'
            # Trim to exact duration
            f'trim=duration={segment_duration},'
            f'setpts=PTS-STARTPTS'
            f'[v{i}];'
        )
        filter_chains.append(chain)
        logger.debug(f"Generated filter chain for video {i+1}/{num_videos}")
    
    # Concatenate all labeled outputs and add subtitles
    concat_part = ''.join(f'[v{i}]' for i in range(num_videos))
    filter_chains.append(
        f'{concat_part}concat=n={num_videos}:v=1:a=0[outv_raw];'
        f'[outv_raw]ass={ass_subtitle_file}[outv]'
    )
    
    filter_complex_str = ''.join(filter_chains)
    audio_input_index = num_videos
    
    cmd = [
        'ffmpeg',
        '-y',
        *inputs,
        '-i', audio_file,
        '-filter_complex', filter_complex_str,
        '-map', '[outv]',
        '-map', f'{audio_input_index}:a',
        '-c:v', 'libx264',
        '-preset', 'veryslow',  # Maximum quality
        '-crf', '18',          # Higher quality
        '-c:a', 'aac',
        '-pix_fmt', 'yuv420p',
        '-r', str(fps),
        '-shortest',
        output_filename
    ]
    
    logger.info("Running FFmpeg command to create video from MP4 files")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                if 'frame=' in output:
                    logger.debug(output.strip())
                else:
                    logger.info(output.strip())
        
        rc = process.poll()
        if rc == 0:
            logger.info("Video creation completed successfully")
        else:
            stderr_output = process.stderr.read() if process.stderr else "No error output available"
            logger.error(f"FFmpeg failed with return code: {rc}")
            logger.error(f"FFmpeg error output: {stderr_output}")
            raise subprocess.CalledProcessError(rc, cmd, stderr_output)
            
    except Exception as e:
        logger.error(f"Error during video creation: {str(e)}")
        raise
    finally:
        # Clean up temporary ASS file
        if os.path.exists(ass_subtitle_file):
            os.remove(ass_subtitle_file)
            logger.debug("Removed temporary ASS subtitle file")
        
        # Log total execution time
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        logger.info(f"Total execution time: {minutes} minutes and {seconds} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from MP4 files with subtitles and speed control")
    parser.add_argument("--input", required=True, help="Input folder containing MP4 files")
    parser.add_argument("--output", help="Output video filename (optional)")
    parser.add_argument("--audio", default="story.mp3", help="Input audio file (default: story.mp3)")
    parser.add_argument("--subtitles", default="output.srt", help="Input subtitle file (default: output.srt)")
    parser.add_argument("--width", type=int, default=1920, help="Output video width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output video height (default: 1080)")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS (default: 60)")
    parser.add_argument("--speed", type=float, default=1.0, 
                       help="Video speed factor: 1.0=normal, 0.5=half speed, 2.0=double speed (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        create_video_from_videos(
            video_folder=args.input,
            audio_file=args.audio,
            output_filename=args.output,
            output_width=args.width,
            output_height=args.height,
            fps=args.fps,
            subtitle_file=args.subtitles,
            speed_factor=args.speed
        )
        logger.info("Video creation process completed.")
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
