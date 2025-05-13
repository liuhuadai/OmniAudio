import os
import subprocess
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from glob import glob
from PIL import Image, ImageOps
from pathlib import Path
from transformers import AutoProcessor
from transformers import AutoModel
import cv2
import torch
import cupy as cp 
from tqdm import tqdm




def pad_to_square(img, padding_color=(0, 0, 0)):
    """
    Pads the input PIL image to make it square while preserving the aspect ratio.
    
    Parameters:
        img (PIL.Image): Input image.
        padding_color (tuple): RGB color tuple for the padding (default is black).
        
    Returns:
        PIL.Image: The square padded image.
    """
    width, height = img.size
    if width == height:
        return img.copy()

    # Determine padding for left/right or top/bottom
    if width > height:
        padding = (0, (width - height) // 2, 0, (width - height) // 2)
    else:
        padding = ((height - width) // 2, 0, (height - width) // 2, 0)
    
    # Expand the image with the specified padding color
    new_img = ImageOps.expand(img, padding, fill=padding_color)
    
    return new_img

def process_single_image(image_path, processor):
    try:
        with Image.open(image_path) as img:
            # Pad image to square
            padded_img = pad_to_square(img)
            return processor(images=[padded_img], return_tensors="np")["pixel_values"]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_frames_directory(frames_dir,processor, fps=8):
    """
    Processes all frame images in a given directory in parallel.
    """
    max_workers = 4
    image_paths = sorted(glob(os.path.join(frames_dir, '*.jpg')))
    if not image_paths:
        raise ValueError(f"No jpg images found in {frames_dir}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_single_image, processor=processor)
        results = list(executor.map(process_func, image_paths))

    # Filter out None results and concatenate
    results = [res for res in results if res is not None]
    if not results:
        raise ValueError("No images were successfully processed.")
    
    return np.concatenate(results, axis=0)

def extract_frames(video_path, output_dir, fps=8, image_format="jpg"):
    """
    Extracts frames from a video and saves them to a specified output directory.

    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory where the extracted frames will be saved.
    - fps: Number of frames per second to extract (default is 8).
    - image_format: Format for the output images (default is png).

    The function uses the following ffmpeg command:
    ffmpeg -i video_path -vf fps=fps output_dir/frame_%04d.image_format
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output file path template, e.g., "output_dir/frame_%04d.png"
    output_template = os.path.join(output_dir, f"frame_%04d.{image_format}")
    
    # Build the ffmpeg command to extract frames
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_template
    ]
    
    try:
        # Run the command and check for errors
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"Frame extraction completed. Frames saved in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print("Error occurred during frame extraction:", e)

def compute_embeddings(eq_videos, model, device="cuda:0"):
    """
    Compute embeddings from a NumPy array of video frames (eq_videos) using the provided model.
    
    Parameters:
    - eq_videos (np.ndarray): The input video frames as a NumPy array.
    - model: The pre-initialized model (should already be moved to the specified device).
    - device (str): The device to perform computation on (default is "cuda:0").
    
    Returns:
    - np.ndarray: The computed embeddings.
    """
    # Convert the numpy array to a torch tensor and move it to the specified device.
    inputs = torch.from_numpy(eq_videos).to(device)
    
    # Compute embeddings without tracking gradients.
    with torch.no_grad():
        embedding = model.get_image_features(inputs)
    
    # Move the embeddings back to CPU and convert to a NumPy array.
    embeddings = embedding.cpu().numpy()
    return embeddings





def equirectangular_to_perspective_numpy(equi_img, fov=120, theta=0, phi=0, height=224, width=224):
    """
    CPU version of equirectangular to perspective conversion using NumPy.
    """
    equi_height, equi_width = equi_img.shape[:2]
    
    # Convert angles to radians
    FOV = np.deg2rad(fov)
    THETA = np.deg2rad(theta)
    PHI = np.deg2rad(phi)
    
    # Create meshgrid for coordinates
    x_range = np.linspace(-1, 1, width)
    y_range = np.linspace(-1, 1, height)
    x_range, y_range = np.meshgrid(x_range, y_range)
    
    # Calculate xyz coordinates
    z = 1 / np.tan(FOV / 2)
    d = np.sqrt(x_range**2 + y_range**2 + z**2)
    
    xyz = np.stack([
        x_range / d,
        y_range / d,
        np.full_like(x_range, z) / d
    ], axis=-1)
    
    # Create rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(PHI), -np.sin(PHI)],
        [0, np.sin(PHI), np.cos(PHI)]
    ])
    
    R_y = np.array([
        [np.cos(THETA), 0, np.sin(THETA)],
        [0, 1, 0],
        [-np.sin(THETA), 0, np.cos(THETA)]
    ])
    
    R = R_x @ R_y
    
    # Rotate coordinates
    xyz = np.einsum('ij,hwj->hwi', R.T, xyz)
    
    # Calculate spherical coordinates
    lon = np.arctan2(xyz[..., 0], xyz[..., 2])
    lat = np.arcsin(xyz[..., 1])
    
    # Convert to pixel coordinates
    lon = ((lon / (2 * np.pi) + 0.5) * equi_width).astype(np.float32)
    lat = ((lat / np.pi + 0.5) * equi_height).astype(np.float32)
    
    # Remap image
    perspective_img = cv2.remap(
        equi_img,
        lon,
        lat,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP
    )
    
    return perspective_img

def equirectangular_to_perspective_cuda(equi_img, fov=120, theta=0, phi=0, height=224, width=224):
    """
    CUDA version of equirectangular to perspective conversion.
    """
    try:
        import cupy as cp
        
        # Convert image to cupy array
        if not isinstance(equi_img, cp.ndarray):
            equi_img = cp.asarray(equi_img)
        
        equi_height, equi_width = equi_img.shape[:2]
        
        # Convert angles to radians
        FOV = cp.deg2rad(fov)
        THETA = cp.deg2rad(theta)
        PHI = cp.deg2rad(phi)
        
        # Create meshgrid for coordinates
        x_range = cp.linspace(-1, 1, width)
        y_range = cp.linspace(-1, 1, height)
        x_range, y_range = cp.meshgrid(x_range, y_range)
        
        # Calculate xyz coordinates
        z = 1 / cp.tan(FOV / 2)
        d = cp.sqrt(x_range**2 + y_range**2 + z**2)
        
        xyz = cp.stack([
            x_range / d,
            y_range / d,
            cp.full_like(x_range, z) / d
        ], axis=-1)
        
        # Create rotation matrices
        R_x = cp.array([
            [1, 0, 0],
            [0, cp.cos(PHI), -cp.sin(PHI)],
            [0, cp.sin(PHI), cp.cos(PHI)]
        ])
        
        R_y = cp.array([
            [cp.cos(THETA), 0, cp.sin(THETA)],
            [0, 1, 0],
            [-cp.sin(THETA), 0, cp.cos(THETA)]
        ])
        
        R = cp.matmul(R_x, R_y)
        
        # Rotate coordinates
        xyz = cp.einsum('ij,hwj->hwi', R.T, xyz)
        
        # Calculate spherical coordinates
        lon = cp.arctan2(xyz[..., 0], xyz[..., 2])
        lat = cp.arcsin(xyz[..., 1])
        
        # Convert to pixel coordinates
        lon = ((lon / (2 * cp.pi) + 0.5) * equi_width).astype(cp.float32)
        lat = ((lat / cp.pi + 0.5) * equi_height).astype(cp.float32)
        
        # Remap image
        perspective_img = cv2.remap(
            cp.asnumpy(equi_img),
            cp.asnumpy(lon),
            cp.asnumpy(lat),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        
        # Clean up GPU memory
        del equi_img, lon, lat, xyz, R, R_x, R_y
        cp.get_default_memory_pool().free_all_blocks()
        
        return perspective_img
    
    except Exception as e:
        #print(f"CUDA processing failed: {str(e)}")
        #print("Falling back to CPU processing...")
        return equirectangular_to_perspective_numpy(equi_img, fov, theta, phi, height, width)

def convert_single_image(args):
    """
    Process a single image with automatic CUDA/CPU selection and error handling.
    """
    img_path, output_path, fov, theta, phi, height, width, use_cuda = args
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image file: {img_path}")
        
        # Choose processing function based on CUDA availability
        if use_cuda:
            front_img = equirectangular_to_perspective_cuda(
                img, fov=fov, theta=theta, phi=phi,
                height=height, width=width
            )
        else:
            front_img = equirectangular_to_perspective_numpy(
                img, fov=fov, theta=theta, phi=phi,
                height=height, width=width
            )
        
        cv2.imwrite(str(output_path), front_img)
        return True, str(img_path)
    except Exception as e:
        return False, f"Error processing {img_path}: {str(e)}"

def convert_equirect_to_frontview(
    input_dir, output_dir, fov=120, theta=0, phi=0,
    height=224, width=224, skip_existing=False,
    img_ext='jpg', max_workers=4, batch_size=16,
    use_cuda=True
):
    """
    Optimized batch processing with automatic CUDA/CPU fallback.
    """
    # Check CUDA availability if requested
    if use_cuda:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            print(f"CUDA initialization failed: {str(e)}")
            print("Falling back to CPU processing...")
            use_cuda = False
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Prepare task list
    tasks = []
    for img_path in input_dir.rglob(f'*.{img_ext}'):
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        if skip_existing and output_path.exists():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((img_path, output_path, fov, theta, phi, height, width, use_cuda))
    
    if not tasks:
        print("No files to process")
        return
    
    # Process images in batches with progress bar
    successful = 0
    failed = 0
    errors = []
    
    with tqdm(total=len(tasks), desc=f"Converting images ({'CUDA' if use_cuda else 'CPU'})") as pbar:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(convert_single_image, batch))
                
                # Process results
                for success, message in results:
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        errors.append(message)
                    pbar.update(1)
    
    # Print summary
    print(f"\nConversion complete:")
    print(f"Successfully processed: {successful}/{len(tasks)} images")
    print(f"Failed: {failed}/{len(tasks)} images")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)

def is_cuda_available():
    """
    Check if CUDA is available and properly initialized.
    """
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception as e:
        print(f"CUDA is not available: {str(e)}")
        return False

def extract_video_features(video_path, feature_type, temp_dir):
    if feature_type == "eqfov":
        name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"Extracting frames from {name}...")
        os.makedirs(os.path.join(temp_dir, "frames"), exist_ok=True)
        frames_eq_path = os.path.join(temp_dir, "frames", name + "_eq")
        extract_frames(video_path, frames_eq_path, fps=8, image_format="jpg")

        frames_fov_path = os.path.join(temp_dir, "frames", name + "_fov")
        convert_equirect_to_frontview(frames_eq_path, frames_fov_path)

        print(f"processing frames...")
        processor = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        eq_videos = process_frames_directory(frames_eq_path, processor, fps=8)
        fov_videos = process_frames_directory(frames_fov_path, processor, fps=8)

        del processor
        torch.cuda.empty_cache()


        device = "cuda:0"
        print(f"Extracting features...")
        model = AutoModel.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        model.eval()
        model.to(device)

        def compute_and_save_embeddings(videos, output_dir):
            parent_dir = os.path.dirname(output_dir)  
            os.makedirs(parent_dir, exist_ok=True)
            embeddings = compute_embeddings(videos, model, device=device)
            print(f"Saving embeddings to {output_dir}")
            np.save(output_dir, embeddings)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    compute_and_save_embeddings, 
                    videos, 
                    output_path
                ) for videos, output_path in [
                    (eq_videos, os.path.join(temp_dir, "metaclip-huge-eq", name + ".npy")),
                    (fov_videos, os.path.join(temp_dir, "metaclip-huge-fov", name + ".npy"))
                ]
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Task failed: {e}")
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

