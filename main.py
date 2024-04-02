import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe drawing utility
mp_drawing = mp.solutions.drawing_utils

def main():

    cap = cv2.VideoCapture("video.mp4")

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = cap.read()

        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and draw poses
        results_pose = pose.process(image)

        # Process the image and detect faces
        results_face = face_detection.process(image)

        if results_pose.pose_landmarks:
            # Draw the pose annotations on the image
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results_face.detections:
            for detection in results_face.detections:
                # Draw the face annotations on the image
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow("Pose and Face Detector", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
