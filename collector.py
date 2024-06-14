import cv2
from ultralytics import YOLO, solutions
import time
import pathlib
from datetime import datetime

def main(video_path):
    # read video, path: video/bvideo.mp4
    cap = cv2.VideoCapture(video_path)
    # makedir with pathlib
    pathlib.Path(video_path.stem).mkdir(parents=True, exist_ok=True)
    # load model
    model = YOLO('yolov8s.pt')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, classes=[0])

        # if there is an object detected in the frame, save frame each 60 frames
        if len(results[0]) > 0:

            if frame_count % 450 == 0:
                print('saved frame: ', frame_count)
                now = datetime.now()
                # convert unix time
                now = time.mktime(now.timetuple())

                cv2.imwrite(f'{video_path.stem}/{int(now)}.jpg', frame)

        frame_count += 1

        # # drawing bounding boxes
        # frame = results[0].plot()

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    videos = pathlib.Path('videos').glob('*.mp4')
    videos = sorted(list(videos))
    for video in videos:
        main(video)