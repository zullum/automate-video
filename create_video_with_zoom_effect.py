import subprocess
import os
import logging
import json
from pathlib import Path
from datetime import datetime
import re
import time
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
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,58,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,20,1
Style: Active,Montserrat,58,&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,110,110,0,0,1,4,0,2,20,20,20,1

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
    output_width: int = 1920,
    output_height: int = 1080,
    fps: int = 60,
    subtitle_file: str = "output.srt",
    zoom_params: dict = None
):
    """
    Create a video from images with smooth continuous zoom effect and subtitles.
    
    Args:
        image_folder: Path to folder containing PNG images
        audio_file: Path to audio file
        output_filename: Output video filename (optional)
        output_width: Output video width
        output_height: Output video height
        fps: Output video FPS
        subtitle_file: Path to subtitle file
        zoom_params: Dictionary containing zoom effect parameters:
                    - range: Zoom range (default: 0.25)
                    - x_offset: Horizontal zoom center offset (default: 0.25)
                    - y_offset: Vertical zoom center offset (default: 0.15)
    """
    # Set default zoom parameters if not provided
    if zoom_params is None:
        zoom_params = {
            'range': 0.25,
            'x_offset': 0.25,
            'y_offset': 0.15
        }
    
    start_time = time.time()
    logger.info(f"Starting video creation process from folder: {image_folder}")
    logger.debug(f"Output dimensions: {output_width}x{output_height}, FPS: {fps}")
    logger.info(f"Using subtitle file: {subtitle_file}")
    logger.info(f"Zoom parameters: range={zoom_params['range']}, "
               f"x_offset={zoom_params['x_offset']}, y_offset={zoom_params['y_offset']}")
    
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
            # Base zoom effect
            f'scale=10*iw:10*ih,'
            f'setsar=1,'
            f"zoompan="
            # Simple smooth zoom effect
            f"z='1+{zoom_params['range']}*sin({t_norm}*PI/2)':"
            f"x='iw*{zoom_params['x_offset']}-(iw*{zoom_params['x_offset']}/zoom)':"
            f"y='ih*{zoom_params['y_offset']}-(ih*{zoom_params['y_offset']}/zoom)':"
            f"d={d_frames}:"
            f"s={output_width*3}x{output_height*3}:"
            f"fps={fps},"
            f'scale={output_width}:{output_height}:flags=lanczos'
            f'[v{i}];'
        )
        filter_chains.append(chain)
        logger.debug(f"Generated transition chain for image {i+1}/{num_images}")
    
    # Create crossfade transitions between consecutive videos
    fade_duration = 2  # Duration of fade in seconds
    fade_frames = int(fade_duration * fps)
    
    # Build the xfade chain differently
    xfade_str = ""
    temp_output = "v0"
    
    for i in range(num_images - 1):
        next_output = f"v{i+1}out"
        if i == num_images - 2:  # Last transition
            next_output = "outv_raw"
            
        xfade_str += (
            f"[{temp_output}][v{i+1}]xfade=transition=fade:"
            f"duration={fade_duration}:offset={frame_duration-fade_duration}[{next_output}];"
        )
        temp_output = next_output
    
    # Add subtitles to the final output
    xfade_str += f"[outv_raw]ass={ass_subtitle_file}[outv]"
    filter_chains.append(xfade_str)
    
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
    parser = argparse.ArgumentParser(description="Create video from images with zoom effect and subtitles")
    parser.add_argument("--input", required=True, help="Input folder containing PNG images")
    parser.add_argument("--output", help="Output video filename (optional)")
    parser.add_argument("--audio", default="story.mp3", help="Input audio file (default: story.mp3)")
    parser.add_argument("--subtitles", default="output.srt", help="Input subtitle file (default: output.srt)")
    parser.add_argument("--width", type=int, default=1920, help="Output video width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output video height (default: 1080)")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS (default: 60)")
    parser.add_argument("--zoom-range", type=float, default=0.25, 
                       help="Zoom range for the effect (default: 0.25). Higher values mean more pronounced zoom.")
    parser.add_argument("--zoom-x-offset", type=float, default=0.25,
                       help="Horizontal zoom center offset (default: 0.25). Range 0-1.")
    parser.add_argument("--zoom-y-offset", type=float, default=0.15,
                       help="Vertical zoom center offset (default: 0.15). Range 0-1.")
    
    args = parser.parse_args()
    
    try:
        # Update the zoompan parameters in the filter chain
        create_video_from_images(
            image_folder=args.input,
            audio_file=args.audio,
            output_filename=args.output,
            output_width=args.width,
            output_height=args.height,
            fps=args.fps,
            subtitle_file=args.subtitles,
            zoom_params={
                'range': args.zoom_range,
                'x_offset': args.zoom_x_offset,
                'y_offset': args.zoom_y_offset
            }
        )
        logger.info("Video creation process completed.")
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
