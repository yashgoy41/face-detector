import cv2
import mediapipe as mp
import time

class FaceDetector():

    def __init__(self, min_detection_confidence=0.5):
        self.detCon = min_detection_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.face = self.mp_face_detection.FaceDetection(self.detCon)
        self.mp_drawing = mp.solutions.drawing_utils

    def findFaces(self, image, draw=True):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face.process(image)
        bboxs = []
        if results.detections:
            # print(results.detections)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    image.flags.writeable = True
                    image = self.fancyDraw(image, bbox)
                    cv2.putText(image, f'{int(detection.score[0] *100)}%',
                                (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                (0, 255, 0), 2)
                    # self.mp_drawing.draw_detection(image, detection, bbox_drawing_spec= self.mp_drawing.DrawingSpec(
                    #     thickness=2, color=(0,255,0)))
        return image, bboxs

    def fancyDraw(self, image, bbox, length=20, thickness=3):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(image, bbox, (0, 255, 0), 1)
        # top left
        cv2.line(image, (x, y), (x+length, y),(0, 255, 0), thickness)
        cv2.line(image, (x, y), (x, y+length), (0, 255, 0), thickness)

        # top right
        cv2.line(image, (x1, y), (x1 - length, y), (0, 255, 0), thickness)
        cv2.line(image, (x1, y), (x1, y + length), (0, 255, 0), thickness)

        # bottom left
        cv2.line(image, (x, y1), (x + length, y1), (0, 255, 0), thickness)
        cv2.line(image, (x, y1), (x, y1 - length), (0, 255, 0), thickness)

        # bottom right
        cv2.line(image, (x1, y1), (x1-length, y1), (0, 255, 0), thickness)
        cv2.line(image, (x1, y1), (x1, y1-length), (0, 255, 0), thickness)
        return image


def main():
    cap = cv2.VideoCapture('Videos/3.mp4')
    cap.set(3,320)
    cap.set(4,240)
    detector = FaceDetector()
    pTime = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        image.flags.writeable = False
        image, bboxs = detector.findFaces(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()


if __name__ == '__main__':
    main()

