from django.shortcuts import render
import mediapipe as mp
import cv2
import mediapipe as mp
import numpy as np
# Create your views here.
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def image_upload(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    ## Setup mediapipe instance

def image(request):
    print("hello")
    file = request.FILES.getlist("image")
    print(file)
    li = []
    for f in file:
        li.append(f)
    img = li[0]
    print(img)
    image_upload(img)
    return render(request,"image_output.html")

def video(request):
    print("hi")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates of right hand
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # get coordinates of left hand

                # Calculate angle
                r_angle = calculate_angle(r_wrist, r_shoulder, r_elbow)

                # Visualize angle
                cv2.putText(image, str(r_angle),
                            tuple(np.multiply(r_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                # Calculate angle
                l_angle = calculate_angle(l_wrist, l_shoulder, l_elbow)

                cv2.putText(image, str(l_angle),
                            tuple(np.multiply(l_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                l_hand = int(0)
                r_hand = int(0)
                if l_angle > 130:
                    l_hand = 1
                elif l_angle > 70 and l_angle < 120:
                    l_hand = 2

                if r_angle > 130:
                    r_hand = 1
                elif r_angle > 70 and r_angle < 120:
                    r_hand = 2

                if l_hand == 1 and r_hand == 1:
                    s = "six"
                    cv2.putText(image, str(s),
                                tuple(np.multiply(l_shoulder, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                                )
                elif l_hand == 1 or r_hand == 1:
                    s = "out"
                    cv2.putText(image, str(s),
                                tuple(np.multiply(l_shoulder, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                                )

                if l_hand == 2 and r_hand == 2:
                    s = "wide"
                    cv2.putText(image, str(s),
                                tuple(np.multiply(l_shoulder, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                                )
                elif l_hand == 2 or r_hand == 2:
                    s = "no ball"
                    cv2.putText(image, str(s),
                                tuple(np.multiply(l_shoulder, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                                )

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    return render(request,"thankyou.html")
def home(request):
    return render(request,"index.html")