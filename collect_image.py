import os
import cv2
import time
import uuid

IMAGE_PATH = r"C:\Users\Ragav U\OneDrive\Desktop\Ragav Folder\Projects\SIGN LANG PROJECT\Collected Images for Training"

labels = [
    'Hello',
    'Yes',
    'No',
    'Thanks',
    'ILoveYou',
    'Please',
    'Help'
]

number_of_images = 200  # increase this

cap = cv2.VideoCapture(0)

for label in labels:

    label_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(label_path, exist_ok=True)

    print(f"\nCollecting images for {label}")
    print("Press SPACE to capture image")
    print("Press Q to skip current class")

    time.sleep(2)

    img_count = 0

    while img_count < number_of_images:

        ret, frame = cap.read()

        if not ret:
            continue

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"{label} : {img_count}/{number_of_images}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("VisionSpeak Dataset Collection", display_frame)

        key = cv2.waitKey(1)

        # SPACE = Capture
        if key == 32:

            image_name = os.path.join(
                label_path,
                f"{label}.{uuid.uuid4()}.jpg"
            )

            cv2.imwrite(image_name, frame)

            img_count += 1

            print(f"Saved: {img_count}")

        # Q = Skip
        elif key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Dataset Collection Complete")