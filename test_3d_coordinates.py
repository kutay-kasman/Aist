import cv2
import mediapipe as mp
import numpy as np

# === Initialize BlazePose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,  # 0: Lite, 1: Full, 2: Heavy → Enables 3D
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# === Load your video ===
cap = cv2.VideoCapture(r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR0_ch01.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = pose.process(image)

    if results.pose_landmarks:
        # Draw pose on original frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # === Extract 3D keypoints ===
        landmarks = results.pose_world_landmarks.landmark  # world_landmarks = 3D (in meters)

        for idx, lm in enumerate(landmarks):
            print(f"Landmark {idx}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}, visibility={lm.visibility:.2f}")

    # Show the frame
    cv2.imshow('BlazePose 3D', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

cap.release()
cv2.destroyAllWindows()

pose_3d = [(lm.x, lm.y, lm.z) for lm in results.pose_world_landmarks.landmark]
np.save("frame_001_pose.npy", np.array(pose_3d))