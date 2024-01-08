import cv2 as cv
import mediapipe as mp

# Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Start camera
try:
    vid = cv.VideoCapture(0)
except Exception as e:
    print(f"Error opening video source: {e}")
    exit(1)

# Mediapipe hands model
with mp_hands.Hands(model_complexity= 0,
                    min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5) as hands:
    
    while True:
        # Read frame
        ret, frame = vid.read()
        if not ret or frame is None: 
            print("Failed to capture frame. Exiting")
            break

        # Process frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Draw landmarks on hands in frame
        if results.multi_hand_landmarks:
            for multi_hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, 
                                          multi_hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        # Mirror camera view
        frame = cv.flip(frame, 1)
        # Display frame
        cv.imshow('frame', frame)

        # Exit loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break

# Resource release and cleanup
vid.release()
cv.destroyAllWindows()
