import torch
import cv2
import numpy as np
import imageio
import os
import logging
import PIL.Image
import warnings
import subprocess

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out PIL debug messages about plugin imports
logging.getLogger('PIL').setLevel(logging.WARNING)

# 1) LOAD MiDaS DEPTH MODEL
def load_midas(model_type="DPT_Large"):
    """
    Loads a MiDaS model from PyTorch Hub.
    
    model_type can be:
    - "DPT_Large" (default, more accurate but larger)
    - "DPT_Hybrid"
    - "MiDaS_small" (faster, less accurate)
    """
    # Load the model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform

# 2) ESTIMATE DEPTH
def estimate_depth(image_bgr, midas, transform):
    """
    Takes a BGR image (from cv2) and returns a normalized depth map (H x W).
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Get original image size
    original_size = image_rgb.shape[:2]
    logger.info(f"Original image size: {original_size}")
    
    try:
        # Apply MiDaS transforms
        input_batch = transform(image_rgb)
        logger.debug(f"Transformed input shape: {input_batch.shape}")
        
        # Add batch dimension if not present
        if len(input_batch.shape) == 3:
            input_batch = input_batch.unsqueeze(0)
        logger.debug(f"Input batch shape after unsqueeze: {input_batch.shape}")
        
        # Move to CPU (since we're not using GPU)
        input_batch = input_batch.cpu()
        
        # Inference
        with torch.no_grad():
            prediction = midas(input_batch)
            logger.debug(f"Raw prediction shape: {prediction.shape}")
            
            # Ensure prediction has the right shape for interpolation
            if len(prediction.shape) == 2:
                prediction = prediction.unsqueeze(0).unsqueeze(0)
            elif len(prediction.shape) == 3:
                prediction = prediction.unsqueeze(0)
            
            logger.debug(f"Prediction shape before interpolation: {prediction.shape}")
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=original_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            logger.debug(f"Final prediction shape: {prediction.shape}")

        # Convert to numpy
        depth = prediction.cpu().numpy()
        
        # Normalize to 0-1 for convenience
        depth_min, depth_max = depth.min(), depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        logger.info(f"Depth map generated successfully, shape: {depth.shape}")
        return depth
        
    except Exception as e:
        logger.error(f"Error in depth estimation: {str(e)}")
        logger.error(f"Error details:", exc_info=True)
        raise

def process_depth_map(depth):
    """
    Process depth map to create more distinct separation between foreground and background.
    """
    # Apply contrast enhancement to depth map
    depth = cv2.GaussianBlur(depth, (5, 5), 0)
    
    # Apply adaptive thresholding to create more distinct layers
    depth_mean = np.mean(depth)
    depth_std = np.std(depth)
    
    # Create foreground mask
    foreground_threshold = depth_mean + depth_std * 0.5
    background_threshold = depth_mean - depth_std * 0.5
    
    # Enhance contrast in mid-range depths
    depth = np.power(depth, 1.5)  # Increase contrast
    
    # Normalize again after contrast enhancement
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Smooth transitions
    depth = cv2.GaussianBlur(depth, (3, 3), 0)
    
    return depth

# 3) WARP IMAGE FOR PARALLAX
def warp_image(image_bgr, depth, shift_x=0, shift_y=0, strength=0.1):
    """
    Warps image based on depth with improved foreground/background separation.
    """
    h, w = image_bgr.shape[:2]
    
    # Calculate larger padding to prevent black edges
    pad_size = int(max(abs(shift_x), abs(shift_y)) * strength * 2.5)
    
    # Create padded image and depth
    padded = cv2.copyMakeBorder(
        image_bgr,
        pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_REFLECT_101
    )
    
    depth_padded = cv2.copyMakeBorder(
        depth,
        pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_REFLECT_101
    )
    
    # Create coordinate matrices
    ph, pw = padded.shape[:2]
    x_indices, y_indices = np.meshgrid(
        np.arange(pw, dtype=np.float32),
        np.arange(ph, dtype=np.float32)
    )
    
    # Create edge falloff
    edge_falloff = create_edge_falloff(depth_padded.shape[:2], 0.3)
    
    # Apply depth-based movement reduction
    depth_threshold = 0.3  # Reduce movement of background
    movement_mask = np.clip((depth_padded - depth_threshold) / (1 - depth_threshold), 0, 1)
    movement_mask = cv2.GaussianBlur(movement_mask, (5, 5), 0)
    
    # Combine edge falloff with movement mask
    final_mask = edge_falloff * movement_mask
    
    # Calculate offsets with masked strength
    x_offsets = shift_x * depth_padded * strength * final_mask
    y_offsets = shift_y * depth_padded * strength * final_mask
    
    # Calculate new positions
    map_x = (x_indices + x_offsets).astype(np.float32)
    map_y = (y_indices + y_offsets).astype(np.float32)
    
    # Remap with high-quality interpolation
    warped = cv2.remap(
        padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    # Crop back to original size
    warped = warped[pad_size:pad_size+h, pad_size:pad_size+w]
    
    return warped

def create_edge_falloff(shape, falloff_strength=0.3):
    """Create a smooth falloff map for depth edges."""
    h, w = shape
    logger.debug(f"Creating edge falloff map for shape: {shape}")
    
    # Create coordinate matrices
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), 
                      np.arange(w, dtype=np.float32), 
                      indexing='ij')
    
    # Calculate distances from each edge
    dist_from_left = x
    dist_from_right = w - 1 - x
    dist_from_top = y
    dist_from_bottom = h - 1 - y
    
    # Find minimum distance to any edge (already in correct shape)
    min_dist = np.minimum.reduce([
        dist_from_left,
        dist_from_right,
        dist_from_top,
        dist_from_bottom
    ])
    
    # Normalize distances
    scale_factor = falloff_strength * min(h, w)
    falloff = min_dist / scale_factor
    falloff = np.clip(falloff, 0, 1)
    
    # Apply sigmoid for smoother transition
    falloff = 1 / (1 + np.exp(-10 * (falloff - 0.5)))
    
    logger.debug(f"Edge falloff map shape: {falloff.shape}")
    return falloff.astype(np.float32)

# 4) GENERATE A LOOPING ANIMATION
def generate_3d_parallax_animation(
    input_image_path,
    output_path="parallax.gif",
    frames=60,
    amplitude=35,
    strength=0.7,
    fps=60,
    model_type="DPT_Large",
    output_format="gif",
    output_width=None,
    output_height=None
):
    """
    Generates a looping parallax animation from a single image.
    
    Args:
        output_format: Either 'gif' or 'mp4'. MP4 provides better quality.
        output_width: Desired output width (optional, defaults to original image width)
        output_height: Desired output height (optional, defaults to original image height)
    """
    logger.info(f"Generating 3D parallax animation in {output_format.upper()} format")
    
    # Load image and setup
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image {input_image_path}")
    
    # Get original dimensions
    orig_height, orig_width = image_bgr.shape[:2]
    
    # Set output dimensions
    if output_width is None:
        output_width = orig_width
    if output_height is None:
        output_height = orig_height
        
    # Resize input image if output dimensions are different
    if (output_width != orig_width) or (output_height != orig_height):
        image_bgr = cv2.resize(image_bgr, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"Resized input image from {orig_width}x{orig_height} to {output_width}x{output_height}")
    
    # Load MiDaS model
    midas, transform = load_midas(model_type)
    
    # Estimate depth
    depth = estimate_depth(image_bgr, midas, transform)
    depth = process_depth_map(depth)
    
    # Setup animation parameters
    height, width = image_bgr.shape[:2]
    max_scale = 1.15
    max_height = int(height * max_scale)
    max_width = int(width * max_scale)
    pad_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    
    frames_list = []
    logger.info("Generating animation frames with enhanced 3D effect...")
    
    for i in range(frames):
        angle = 2 * np.pi * i / frames
        
        # Smooth movement patterns
        shift_x = amplitude * np.sin(angle)
        shift_y = amplitude * 0.15 * np.sin(angle * 1.5)
        
        # Gentler zoom effect
        zoom_factor = 0.06
        scale = 1.0 + zoom_factor * (1 + np.cos(angle)) / 2
        
        # Warp image with improved depth handling
        warped = warp_image(image_bgr, depth, shift_x, shift_y, strength)
        
        # Apply zoom and center
        new_height = int(height * scale)
        new_width = int(width * scale)
        zoomed = cv2.resize(warped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create frame with padding
        frame = pad_frame.copy()
        y_offset = (max_height - new_height) // 2
        x_offset = (max_width - new_width) // 2
        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = zoomed
        
        # Crop to original size
        start_y = (max_height - height) // 2
        start_x = (max_width - width) // 2
        frame = frame[start_y:start_y+height, start_x:start_x+width]
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame_rgb)
        
        if i % 10 == 0:
            logger.debug(f"Generated frame {i+1}/{frames}")
    
    logger.info(f"Saving animation in {output_format.upper()} format...")
    
    if output_format.lower() == 'mp4':
        # Save as MP4 with high quality settings
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        temp_frames_dir = "temp_frames"
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        try:
            # Save frames as high-quality PNG files
            for i, frame in enumerate(frames_list):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
                PIL.Image.fromarray(frame).save(frame_path, quality=100)
            
            # Use FFmpeg to create high-quality MP4
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_frames_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-preset', 'veryslow',  # Highest quality encoding
                '-crf', '17',           # High quality (0-51, lower is better)
                '-pix_fmt', 'yuv420p',  # Ensure compatibility
                '-movflags', '+faststart',  # Enable streaming
                '-profile:v', 'high',    # High profile for better quality
                '-tune', 'animation',    # Optimize for animation content
                '-filter:v', 'fps=fps=' + str(fps),  # Ensure consistent framerate
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"MP4 animation saved to {output_path}")
            
        finally:
            # Clean up temporary files
            import shutil
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
                
    else:
        # Save as GIF
        imageio.mimsave(
            output_path,
            frames_list,
            fps=fps,
            optimize=True,
            subrectangles=True
        )
        logger.info(f"GIF animation saved to {output_path}")

def batch_process_images(
    input_folder,
    output_folder,
    frames=60,
    amplitude=35,
    strength=0.7,
    fps=60,
    model_type="DPT_Large",
    output_format="mp4",
    output_width=None,
    output_height=None
):
    """
    Process all images in the input folder and create 3D parallax animations.
    
    Args:
        output_format: Either 'gif' or 'mp4'. MP4 provides better quality.
        output_width: Desired output width (optional)
        output_height: Desired output height (optional)
    """
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Batch processing images to {output_format.upper()} format")
    
    # Load MiDaS model once for all images
    logger.info("Loading MiDaS model...")
    midas, transform = load_midas(model_type)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        logger.warning(f"No image files found in {input_folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for i, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, image_file)
        output_filename = os.path.splitext(image_file)[0] + f'.{output_format}'
        output_path = os.path.join(output_folder, output_filename)
        
        logger.info(f"Processing image {i}/{len(image_files)}: {image_file}")
        
        try:
            generate_3d_parallax_animation(
                input_image_path=input_path,
                output_path=output_path,
                frames=frames,
                amplitude=amplitude,
                strength=strength,
                fps=fps,
                model_type=model_type,
                output_format=output_format,
                output_width=output_width,
                output_height=output_height
            )
            logger.info(f"Successfully processed {image_file}")
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")
            continue
    
    logger.info("Batch processing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D parallax animations from images")
    parser.add_argument("--input", required=True, help="Input image path or folder")
    parser.add_argument("--output", required=True, help="Output animation path or folder")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames")
    parser.add_argument("--amplitude", type=float, default=35, help="Movement amplitude")
    parser.add_argument("--strength", type=float, default=0.7, help="3D effect strength")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--model", default="DPT_Large", help="MiDaS model type")
    parser.add_argument("--format", default="mp4", choices=['gif', 'mp4'], help="Output format (gif or mp4)")
    parser.add_argument("--width", type=int, help="Output width (optional, default: same as input)")
    parser.add_argument("--height", type=int, help="Output height (optional, default: same as input)")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        generate_3d_parallax_animation(
            input_image_path=args.input,
            output_path=args.output,
            frames=args.frames,
            amplitude=args.amplitude,
            strength=args.strength,
            fps=args.fps,
            model_type=args.model,
            output_format=args.format,
            output_width=args.width,
            output_height=args.height
        )
    else:
        batch_process_images(
            input_folder=args.input,
            output_folder=args.output,
            frames=args.frames,
            amplitude=args.amplitude,
            strength=args.strength,
            fps=args.fps,
            model_type=args.model,
            output_format=args.format,
            output_width=args.width,
            output_height=args.height
        )
