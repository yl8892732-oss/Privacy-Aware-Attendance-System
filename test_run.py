import cv2
import numpy as np
import time
from feature_extraction import get_face_embedding


def test_my_module():
    print("Starting camera, please look at the lens...")
    # 0 represents the default camera. If you have an external or virtual camera, you might need to change it to 1 or 2.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera! Please check if it is being used by another program (e.g., WeChat, Zoom).")
        return

    # Give the camera 1 second to warm up, skipping the initial black frames.
    time.sleep(1)

    print("Camera is ready.")
    print("👉 Press 's' on the keyboard to print the current feature vector.")
    print("👉 Press 'q' on the keyboard to exit the test.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        try:
            # Invoke your core extraction function
            embedding = get_face_embedding(frame)

            # Check if extraction was successful (not all zeros)
            if embedding.shape[0] > 0 and not np.all(embedding == 0):
                status_text = "Success! Face Detected"
                color = (0, 255, 0)  # Green
            else:
                status_text = "No Face Detected"
                color = (0, 0, 255)  # Red

            # Overlay status text on the video frame
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        except Exception as e:
            cv2.putText(frame, f"Error: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Pop up a window to display the real-time video frame
        cv2.imshow("Role 2 - Model Test", frame)

        # Capture keyboard operations
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press q to exit
            print("Test finished.")
            break
        elif key == ord('s'):  # Press s to print data
            print("\n--- Manual Extraction Triggered ---")
            if not np.all(embedding == 0):
                print(f"✅ Success! Feature vector dimensions: {embedding.shape}")
                print(f"First 5 elements display: {embedding[:5]}")
            else:
                print("⚠️ No face detected in the current frame, output is all zeros.")

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_my_module()