import cv2
import os
from pathlib import Path

def segment_video_into_30frames(video_path, output_dir=None, segment_length=30):
    """
    Segment a video into chunks of 30 frames each.
    Each segment is saved as a separate video file for manual labeling.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save segments (default: segments folder next to video)
        segment_length: Number of frames per segment (default: 30)
    """
    # Validate video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Set output directory
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        video_name = Path(video_path).stem
        output_dir = os.path.join(video_dir, f"{video_name}_segments")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Estimated Segments: {total_frames // segment_length}")
    print(f"   Output Directory: {output_dir}\n")
    
    # Video writer for current segment
    segment_index = 0
    frame_count = 0
    out = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Save last segment if it has frames
            if out is not None and frame_count > 0:
                out.release()
                print(f"Segment {segment_index:04d} saved ({frame_count} frames)")
            break
        
        # Start new segment if needed
        if frame_count == 0:
            segment_filename = os.path.join(output_dir, f"segment_{segment_index:04d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(segment_filename, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"Error: Could not create video writer for segment {segment_index}")
                break
        
        # Write frame to current segment
        out.write(frame)
        frame_count += 1
        
        # Complete segment when we reach segment_length
        if frame_count == segment_length:
            out.release()
            print(f"Segment {segment_index:04d} saved ({frame_count} frames)")
            segment_index += 1
            frame_count = 0
            out = None
    
    cap.release()
    if out is not None:
        out.release()
    
    print(f"\nSegmentation complete!")
    print(f"   Total segments created: {segment_index + (1 if frame_count > 0 else 0)}")
    print(f"   Segments saved in: {output_dir}")

if __name__ == "__main__":
    # Video path
    video_path = r"E:\coding\datasetPreparation\NiAD_Large Videos\Training\03-Stxavier-1 - Camera 1 - 03-15-20 20.19.01_XVID.avi"
    
    # Segment the video
    segment_video_into_30frames(video_path, segment_length=30)

