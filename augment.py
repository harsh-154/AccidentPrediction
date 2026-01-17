import cv2
import os
import numpy as np

accident_dir = r"E:\coding\datasetPreparation\NiAD_Large Videos\Training\Accident"
output_dir = os.path.join(accident_dir, "augmented")
os.makedirs(output_dir, exist_ok=True)

def increase_brightness(frame, value=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = cv2.min(v, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def horizontal_shift(frame, shift_value=30):
    rows, cols = frame.shape[:2]
    M = np.float32([[1, 0, shift_value], [0, 1, 0]])  # horizontal translation
    shifted = cv2.warpAffine(frame, M, (cols, rows))
    return shifted

def process_video(file_path):
    filename = os.path.basename(file_path)
    name = filename.replace(".mp4","")

    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_shift_right = cv2.VideoWriter(os.path.join(output_dir, f"{name}_shift_right.mp4"), fourcc, fps, (w,h))
    out_shift_left = cv2.VideoWriter(os.path.join(output_dir, f"{name}_shift_left.mp4"), fourcc, fps, (w,h))
    out_bright = cv2.VideoWriter(os.path.join(output_dir, f"{name}_bright.mp4"), fourcc, fps, (w,h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Horizontal shifts
        shifted_right = horizontal_shift(frame, shift_value=30)
        shifted_left = horizontal_shift(frame, shift_value=-30)

        # Brightness enhancement
        bright = increase_brightness(frame, value=40)

        out_shift_right.write(shifted_right)
        out_shift_left.write(shifted_left)
        out_bright.write(bright)

    cap.release()
    out_shift_right.release()
    out_shift_left.release()
    out_bright.release()
    print(f"✅ Augmented: {filename}")

for file in os.listdir(accident_dir):
    if file.endswith(".mp4"):
        process_video(os.path.join(accident_dir, file))

print("\nHorizontal shift + Brightness augmentation completed!")
