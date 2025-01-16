import subprocess
import os
import logging
import json
from pathlib import Path
from datetime import datetime
import random
import math

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

def create_video_from_images(
    image_folder: str,
    audio_file: str = "story.mp3",
    output_filename: str = None,
    output_width: int = 1408,
    output_height: int = 768,
    fps: int = 60
):
    """
    Create a video from images with smooth continuous zoom effect.
    """
    logger.info(f"Starting video creation process from folder: {image_folder}")
    logger.debug(f"Output dimensions: {output_width}x{output_height}, FPS: {fps}")
    
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
            f'zoompan='
            # Use larger internal zoom for smoothness, but scale down the effect
            f"z='1+0.2*sin({t_norm}*PI/2)':"  # Internal zoom range 1.0 to 1.2
            # Fixed offset position for stable movement
            f"x='iw*0.25-(iw*0.25/zoom)':"
            f"y='ih*0.15-(ih*0.15/zoom)':"
            f"d={d_frames}:"
            # Scale the output size up to allow for smoother interpolation
            f"s={output_width*3}x{output_height*3}:"  # Increased intermediate size for smoother scaling
            f"fps={fps},"
            # Scale down to final size, which will reduce the apparent zoom effect
            f'scale={output_width}:{output_height}:flags=lanczos,'  # Added lanczos scaling for smoother result
            # Smooth transitions
            f'fade=t=in:st=0:d=1:alpha=1,'
            f'fade=t=out:st={max(frame_duration - 1, 0)}:d=1:alpha=1,'
            f'trim=duration={frame_duration},'
            f'setpts=PTS-STARTPTS'
            f'[v{i}];'
        )
        filter_chains.append(chain)
        logger.debug(f"Generated smooth zoom chain for image {i+1}/{num_images}")
    
    # Concatenate all labeled outputs
    concat_part = ''.join(f'[v{i}]' for i in range(num_images))
    filter_chains.append(f'{concat_part}concat=n={num_images}:v=1:a=0[outv]')
    
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
    
    logger.info("Running FFmpeg command with smooth zoom effect")
    
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
    
    logger.info(f"Video saved as: {output_filename}")

if __name__ == "__main__":
    image_folder = str(Path("generated_images").absolute())
    try:
        create_video_from_images(image_folder, audio_file="story.mp3")
        logger.info("Video creation process completed.")
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
