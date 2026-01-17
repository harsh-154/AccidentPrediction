import cv2
import os

def segment_video(video_path, out_dir, segment_length=30):
    cap = cv2.VideoCapture(video_path)
    segment_index = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        if len(frames) == segment_length:
            seg_folder = os.path.join(out_dir, f"segment_{segment_index}")
            os.makedirs(seg_folder, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(seg_folder, f"frame_{i}.jpg"), f)
            frames = []
            segment_index += 1

    cap.release()

def process_dataset(src_root, dst_root):
    for cls in ["normal", "accident"]:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for video in os.listdir(src_dir):
            video_path = os.path.join(src_dir, video)
            video_name = video.split('.')[0]
            out_dir = os.path.join(dst_root, cls, video_name)
            os.makedirs(out_dir, exist_ok=True)
            segment_video(video_path, out_dir)

process_dataset("dataset/final_dataset/train", "dataset/segments/train")
process_dataset("dataset/final_dataset/test", "dataset/segments/test")

print("✅ All videos segmented into 30-frame clips!")
