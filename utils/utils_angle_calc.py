import numpy as np

def compute_angles_from_landmarks(landmarks):
    def angle_between(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    pairs = [
        (11, 13, 15), (12, 14, 16),
        (23, 25, 27), (24, 26, 28),
        (11, 23, 25), (12, 24, 26),
        (23, 24, 12), (11, 12, 24)
    ]

    angles = []
    for a, b, c in pairs:
        if max(a, b, c) >= len(landmarks):
            angles.append(0.0)
        else:
            angles.append(angle_between(np.array(landmarks[a]), np.array(landmarks[b]), np.array(landmarks[c])))
    return np.array(angles)
