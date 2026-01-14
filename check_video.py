import cv2

video_path = "D:/X/VideoRAG/movie.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

print("=" * 50)
print("Video Info: movie.mp4")
print("=" * 50)
print(f"Duration: {duration:.1f} sec ({duration/60:.1f} min)")
print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Resolution: {width} x {height}")
print(f"Frames at 1sec interval: {int(duration)} frames")
