import subprocess
import os
import logging
import json
from pathlib import Path
from datetime import datetime
import re
import time

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
Style: Default,Montserrat,55,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,20,1
Style: Active,Montserrat,62,&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,110,110,0,0,1,4,0,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''

    # Read SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Parse SRT content
    srt_blocks = re.split(r'\n\n+', srt_content.strip())
    ass_events = []

    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3:  # Valid subtitle block
            # Parse timing
            time_line = lines[1]
            start_time, end_time = time_line.split(' --> ')
            
            # Convert text to ASS format with word-level styling
            text = ' '.join(lines[2:])
            words = text.upper().split()  # Convert to uppercase
            
            # Calculate timing for each word
            total_duration = parse_srt_time(end_time) - parse_srt_time(start_time)
            word_duration = total_duration / len(words)
            
            # Create ASS events for each word
            for i, word in enumerate(words):
                word_start = parse_srt_time(start_time) + (i * word_duration)
                word_end = word_start + word_duration
                
                # Format times for ASS (h:mm:ss.cc)
                start_str = f"{int(word_start//3600):d}:{int((word_start%3600)//60):02d}:{word_start%60:05.2f}"
                end_str = f"{int(word_end//3600):d}:{int((word_end%3600)//60):02d}:{word_end%60:05.2f}"
                
                # Create animation effect for the active word
                other_words = words.copy()
                other_words[i] = f"{{\\rStyle(Active)}}{word}{{\\rStyle(Default)}}"
                styled_line = ' '.join(other_words)
                
                ass_events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{styled_line}")

    # Write ASS file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        for event in ass_events:
            f.write(event + '\n')
    
    logger.info(f"Created ASS subtitle file: {output_file}")
    return output_file

def create_video_from_images(
    image_folder: str,
    audio_file: str = "story.mp3",
    output_filename: str = None,
    output_width: int = 1408,
    output_height: int = 768,
    fps: int = 60,
    subtitle_file: str = "output.srt"
):
    """
    Create a video from images with smooth continuous zoom effect and subtitles.
    """
    start_time = time.time()
    logger.info(f"Starting video creation process from folder: {image_folder}")
    logger.debug(f"Output dimensions: {output_width}x{output_height}, FPS: {fps}")
    logger.info(f"Using subtitle file: {subtitle_file}")
    
    # Convert SRT to ASS with enhanced styling
    ass_subtitle_file = "temp_subtitles.ass"
    create_ass_subtitles(subtitle_file, ass_subtitle_file)
    logger.info("Created enhanced ASS subtitles with word-level animations")
    
    # Get audio duration
    audio_duration = get_audio_duration(audio_file)
    logger.info(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Gather PNG images
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
    )
    if not image_files:
        logger.error("No PNG images found in the specified folder.")
        return
    
    num_images = len(image_files)
    frame_duration = audio_duration / num_images
    logger.info(f"Found {num_images} images; each will display for ~{frame_duration:.2f} seconds")
    
    # Build output filename if not provided
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_video_{timestamp}.mp4"
    
    # Prepare FFmpeg inputs
    inputs = []
    for img in image_files:
        image_path = os.path.join(image_folder, img)
        inputs.extend(['-loop', '1', '-i', image_path])
    
    d_frames = int(frame_duration * fps)
    logger.debug(f"Frames per image segment: {d_frames}")
    
    filter_chains = []
    for i in range(num_images):
        # Calculate time variable for zoom effect
        t_norm = f"(on/{d_frames})"  # Normalized time from 0 to 1
        
        chain = (
            f'[{i}:v]'
            f'format=yuv420p,'
            # Scale up image significantly for smooth zoom
            f'scale=10*iw:10*ih,'
            f'setsar=1,'
            # Zoompan with smooth continuous zoom
            f"zoompan="
            f"z='1+0.25*sin({t_norm}*PI/2)':"  # Internal zoom range 1.0 to 1.2
            f"x='iw*0.25-(iw*0.25/zoom)':"
            f"y='ih*0.15-(ih*0.15/zoom)':"
            f"d={d_frames}:"
            f"s={output_width*3}x{output_height*3}:"
            f"fps={fps},"
            # Scale down to final size
            f'scale={output_width}:{output_height}:flags=lanczos,'
            # Smooth transitions
            f'fade=t=in:st=0:d=1:alpha=1,'
            f'fade=t=out:st={max(frame_duration - 1, 0)}:d=1:alpha=1,'
            f'trim=duration={frame_duration},'
            f'setpts=PTS-STARTPTS'
            f'[v{i}];'
        )
        filter_chains.append(chain)
        logger.debug(f"Generated smooth zoom chain for image {i+1}/{num_images}")
    
    # Concatenate all labeled outputs and add subtitles
    concat_part = ''.join(f'[v{i}]' for i in range(num_images))
    filter_chains.append(
        f'{concat_part}concat=n={num_images}:v=1:a=0[outv_raw];'
        f'[outv_raw]ass={ass_subtitle_file}[outv]'
    )
    
    filter_complex_str = ''.join(filter_chains)
    audio_input_index = num_images
    
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
    
    logger.info("Running FFmpeg command with smooth zoom effect and subtitles")
    
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
    image_folder = str(Path("generated_images").absolute())
    try:
        create_video_from_images(image_folder, audio_file="story.mp3")
        logger.info("Video creation process completed.")
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
