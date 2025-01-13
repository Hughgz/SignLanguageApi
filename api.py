from flask import Flask
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'I Love U', 26: 'Phắc Du', 27: 'Moawwww', 28: 'Anh Loc', 29: 'Thanh Tai', 30: 'Minh Hieu'
}


@app.route('/')
def index():
    return "Hand sign detection backend"

def preprocess_landmarks(landmarks, W, H):
    """
    Tiền xử lý tọa độ landmark để chuẩn hóa.
    landmarks: landmark từ MediaPipe
    W, H: Chiều rộng và chiều cao khung hình
    """
    data_aux = []
    x_ = []
    y_ = []

    # Trích xuất tọa độ landmark
    for i in range(len(landmarks.landmark)):
        x = landmarks.landmark[i].x
        y = landmarks.landmark[i].y
        x_.append(x)
        y_.append(y)

    # Chuẩn hóa tọa độ landmark
    for i in range(len(landmarks.landmark)):
        x = landmarks.landmark[i].x
        y = landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))

    # Kiểm tra số lượng đặc trưng
    if len(data_aux) != 42:
        raise ValueError(f"Expected 42 features, but got {len(data_aux)}")

    return data_aux

@socketio.on('stream')
def stream_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmark lên khung hình
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                try:
                    # Tiền xử lý dữ liệu
                    data_aux = preprocess_landmarks(hand_landmarks, W, H)

                    # Dự đoán
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Vẽ kết quả lên khung hình
                    x1 = int(min([landmark.x for landmark in hand_landmarks.landmark]) * W) - 10
                    y1 = int(min([landmark.y for landmark in hand_landmarks.landmark]) * H) - 10
                    x2 = int(max([landmark.x for landmark in hand_landmarks.landmark]) * W) + 10
                    y2 = int(max([landmark.y for landmark in hand_landmarks.landmark]) * H) + 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except ValueError as e:
                    print("Preprocessing error:", e)

        # Encode frame as Base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send frame to client
        socketio.emit('frame', {'image': frame_base64})
    cap.release()