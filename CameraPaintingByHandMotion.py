import cv2
import numpy as np
import mediapipe as mp
import os

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Mediapipe and OpenCV modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Colors dictionary for easy color selection
colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255)
}


def init_canvas(width=640, height=480):
    """Initialize a blank canvas for drawing."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_line_on_canvas(canvas, start_point, end_point, color, thickness=5):
    """Draw a line on the canvas between two points."""
    cv2.line(canvas, start_point, end_point, color, thickness)
    return canvas


def process_frame(frame):
    """Process the frame to detect and draw hand landmarks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    return result


def display_frames(frame, canvas, painting_mode):
    """Display the camera feed and the canvas."""
    mode_text = "Painting ON" if painting_mode else "Painting OFF"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Painting', canvas)


def change_color(key, current_color):
    """Change the drawing color based on key pressed."""
    if key == ord('r'):
        return 'red'
    elif key == ord('g'):
        return 'green'
    elif key == ord('b'):
        return 'blue'
    elif key == ord('y'):
        return 'yellow'
    return current_color


def is_finger_fully_extended(landmarks, finger_tip, finger_dip, finger_mcp):
    """Check if a finger is fully extended based on its landmarks."""
    return landmarks[finger_tip].y < landmarks[finger_dip].y < landmarks[finger_mcp].y


def count_extended_fingers(landmarks):
    """Count how many fingers are fully extended."""
    extended_fingers = 0
    fingers = [(8, 7, 5),  # Index finger (tip, dip, MCP)
               (12, 11, 9),  # Middle finger (tip, dip, MCP)
               (16, 15, 13),  # Ring finger (tip, dip, MCP)
               (20, 19, 17)]  # Pinky finger (tip, dip, MCP)

    for tip, dip, mcp in fingers:
        if is_finger_fully_extended(landmarks, tip, dip, mcp):
            extended_fingers += 1

    return extended_fingers


def detect_painting_mode(landmarks):
    """Determine if the hand is in painting mode based on finger extension."""
    extended_fingers = count_extended_fingers(landmarks)
    index_finger_fully_extended = is_finger_fully_extended(landmarks, 8, 7, 5)

    # Painting mode is on if exactly one finger (index finger) is fully extended
    return extended_fingers == 1 and index_finger_fully_extended


def is_valid_landmark(landmark, frame_shape):
    """Check if the landmark is within valid bounds of the frame."""
    x, y = int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])
    return 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]


def main():
    cap = cv2.VideoCapture(0)
    canvas = init_canvas()
    selected_color = 'red'
    prev_position = None  # To store the previous position of the finger
    painting_mode = False  # To toggle painting mode on/off

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = process_frame(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Determine if painting mode should be on based on finger positions
                painting_mode = detect_painting_mode(landmarks)

                # Get the tip of the index finger (landmark 8)
                index_finger_tip = landmarks[8]

                # Check if the landmark is within valid bounds
                if is_valid_landmark(index_finger_tip, frame.shape):
                    current_position = (int(index_finger_tip.x * frame.shape[1]),
                                        int(index_finger_tip.y * frame.shape[0]))

                    if prev_position and painting_mode:
                        # Draw a line from the previous position to the current position
                        canvas = draw_line_on_canvas(canvas, prev_position, current_position, colors[selected_color])

                    prev_position = current_position

                    # Draw landmarks on the frame for visualization
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                else:
                    prev_position = None
        else:
            # Reset the previous position if no hand is detected
            prev_position = None

        display_frames(frame, canvas, painting_mode)

        key = cv2.waitKey(1)
        if key == 27:  # Escape key to exit
            break
        selected_color = change_color(key, selected_color)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
