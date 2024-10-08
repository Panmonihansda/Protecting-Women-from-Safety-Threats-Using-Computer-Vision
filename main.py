import cv2
import time
import mediapipe as mp
from a_person import detect_person
from b_gender import classify_gender
from d_track import CentroidTracker
from e_alert import is_female_surrounded
from h_pose import detect_action
from f_alert import send_telegram_alert
from g_facial import classify_face, draw_selected_landmarks  # Import the functions

# Initialize video capture and tracker
path1 = r'C:\Users\91702\final_project\vid1.mp4'
path2 = r'C:\Users\91702\final_project\vid2.mp4'
webcam = cv2.VideoCapture(path1)
tracker = CentroidTracker()

mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
if not webcam.isOpened():
    print("Could not open video")
    exit()

try:
    skip_frame = 2
    frame_count = -1

    while True:
        status, frame = webcam.read()
        if not status:
            print("Failed to read frame from video")
            break

        frame_count += 1
        if frame_count % skip_frame != 0:
            continue

        # Detect persons in the frame
        person_boxes = detect_person(frame)
        n = len(person_boxes)  # stores the number of persons

        # Reset gender counts for the current frame
        male_count = 0
        female_count = 0
        mbbox = []

        # Update tracker with detected person bounding boxes
        objects = tracker.update(person_boxes)
        print(f"Number of detected persons: {len(person_boxes)}")
        print(f"Number of tracked objects: {len(objects)}")

        for i, (objectID, centroid) in enumerate(objects.items()):
            if objectID < len(person_boxes):
                x1, y1, x2, y2 = map(int, person_boxes[i])  # Ensure bounding box values are integers
                person_image = frame[y1:y2, x1:x2]

                # Check if the person region is sufficiently large to perform detection
                if person_image.size > 0:
                    # Detect and classify facial expression
                    results = mp_holistic.process(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
                    
                    if results.face_landmarks:
                        face_class = classify_face(results.face_landmarks)  # Use the imported classify_face function

                        # Perform gender classification
                        gender_label = classify_gender(person_image)
                        print(f"Gender for objectID {objectID}: {gender_label}")

                        if gender_label:
                            # Update counts based on gender
                            if 'male' in gender_label:  # Check for 'Male' in the label
                                male_count += 1
                                mbbox.append(person_image)
                            elif 'female' in gender_label: 
                                female_bbox = person_image                                
                                female_count += 1

                            # Detect and classify pose
                            pose_action = detect_action(results.pose_landmarks)

                            # Combine all detected features into the label
                            label = f'ID {objectID}: {gender_label}, {face_class}, {pose_action}'

                            # Draw facial landmarks
                            draw_selected_landmarks(person_image, results.face_landmarks)  # Use the imported draw_selected_landmarks function

                            # Alert condition: Female detected alone at night
                            if n == 1 and 'female' in gender_label :#and (time.localtime().tm_hour >= 18 or time.localtime().tm_hour < 6):
                                send_telegram_alert(frame, "Female detected alone at night!")
                                print("Alert sent: Female detected alone at night.")
                        else:
                            label = f'ID {objectID}: Unknown'
                    else:
                        label = f'ID {objectID}: Person'

                    # Annotate the frame
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, tuple(map(int, centroid)), 4, (255, 0, 0), -1)
                else:
                    print(f"Warning: No bounding box for object ID {objectID}")
                    
        # This particular part detects and alerts if the female is surrounded by males.
        if female_count == 1 and n > 2 and (face_class == 'Fear' or face_class == 'Distress'):
            if is_female_surrounded(female_bbox, mbbox):
                send_telegram_alert(frame, "Female surrounded by men, potential danger detected!")
                print("Alert sent: Female surrounded by men.")
        
        # Display the counts of males, females, and persons for the current frame
        count_text = f'Males: {male_count}  Females: {female_count}  Total Persons: {n}'
        cv2.putText(frame, count_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display the annotated frame
        cv2.imshow("Webcam/Video Feed", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()