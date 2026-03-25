import cv2
import glob
import os

def images_to_video(image_pattern, output_path, fps=30, quality='medium'):
    """
    Convert a sequence of images matching a glob pattern to an MP4 video.
    
    Args:
        image_pattern (str): Glob pattern to match images (e.g., "folder/image*.png")
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video (default: 30)
        quality (str): Video quality - 'high', 'medium', or 'low' (default: 'medium')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get list of image files using glob pattern
        images = sorted(glob.glob(image_pattern))
        if not images:
            print(f"No images found matching pattern: {image_pattern}")
            return False

        # Read the first image to get dimensions
        frame = cv2.imread(images[0])
        if frame is None:
            print(f"Failed to read first image: {images[0]}")
            return False
        
        height, width, layers = frame.shape

        # Quality settings mapping to different bitrates (bits per second)
        quality_params = {
            "superhigh" : 30000000, # 30 Mbps
            'high': 20000000,    # 20 Mbps
            'medium': 10000000,   # 10 Mbps
            'low': 5000000      # 5 Mbps
        }
        
        bitrate = quality_params.get(quality, quality_params['medium'])
        
        # Create a temporary file path for initial video
        temp_output = output_path + '_temp.mp4'
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        # Write each image to the video
        total_images = len(images)
        for i, image_path in enumerate(images, 1):
            frame = cv2.imread(image_path)
            if frame is not None:
                video.write(frame)
                print(f"Processing image {i}/{total_images}: {image_path}", end='\r')
            else:
                print(f"\nWarning: Could not read image {image_path}")

        # Release the video writer
        video.release()

        # Use ffmpeg for final compression with specified bitrate
        ffmpeg_command = f'ffmpeg -i {temp_output} -b:v {bitrate} -maxrate {bitrate} -bufsize {bitrate//2} {output_path}'
        os.system(ffmpeg_command)
        
        # Remove temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)

        print(f"\nVideo successfully created at: {output_path}")
        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    keywords = {
        "amp": 'high',
        "frame": 'high',
        "phase": 'superhigh',
        "raw_amp": 'high',
        "raw_phase": 'high'
    }
    
    folder = "result_materials_randn3phase_20k_128x128_tv_12kgs_2.5e-2m_700_betterloss"

    # Process each keyword with its specified quality
    for keyword, quality in keywords.items():
        image_pattern = f"./{folder}/{keyword}*.png"
        output_path = f"{folder}_{keyword}.mp4"
        
        # Create video with 30 fps and specified quality
        images_to_video(image_pattern, output_path, fps=30, quality=quality)