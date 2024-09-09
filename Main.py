import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import tempfile
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def extract_face_encoding(face_landmarks):
    face_encoding = []
    for landmark in face_landmarks.landmark:
        face_encoding.extend([landmark.x, landmark.y, landmark.z])
    return np.array(face_encoding)


def register_face(name, video_file):
    face_encodings = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
       
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return False
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0: 
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_encoding = extract_face_encoding(face_landmarks)
                    face_encodings.append(face_encoding)
                    
                   
                    annotated_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                    st.image(annotated_frame, channels="BGR", use_column_width=True)
            else:
                st.warning(f"No face detected in frame {frame_count}")
    
        cap.release()
        os.unlink(video_path)
    
    if face_encodings:
        face_encodings = np.array(face_encodings)
        average_encoding = np.mean(face_encodings, axis=0)
        std_deviation = np.std(face_encodings, axis=0)
        
        
        if not os.path.exists('face_encodings'):
            os.makedirs('face_encodings')
        
        with open(f'face_encodings/{name}.pkl', 'wb') as f:
            pickle.dump((average_encoding, std_deviation), f)
        
        st.success(f"Registered {len(face_encodings)} face encodings for {name}")
        return True
    else:
        st.error(f"No faces detected in the entire video. Processed {frame_count} frames.")
        return False


def load_registered_faces():
    registered_faces = {}
    if os.path.exists('face_encodings'):
        for filename in os.listdir('face_encodings'):
            if filename.endswith('.pkl'):
                name = os.path.splitext(filename)[0]
                with open(f'face_encodings/{filename}', 'rb') as f:
                    registered_faces[name] = pickle.load(f)
    return registered_faces

def recognize_face(frame, registered_faces):
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_encoding = extract_face_encoding(face_landmarks)
                
                
                similarities = {}
                for name, (registered_encoding, std_deviation) in registered_faces.items():
                    similarity = cosine_similarity([face_encoding], [registered_encoding])[0][0]
                    
                    
                    normalized_diff = np.abs(face_encoding - registered_encoding) / (std_deviation + 1e-6)
                    if np.mean(normalized_diff) > 2:  
                        similarity = 0
                    
                    similarities[name] = similarity
                
                best_match = max(similarities, key=similarities.get)
                if similarities[best_match] > 0.85:
                    return best_match
    
    return "Unknown"


def live_recognition():
    st.header("Live Face Recognition")
    
    registered_faces = load_registered_faces()
    if not registered_faces:
        st.warning("No faces registered yet. Please register faces before recognition.")
        return

    st.write("Registered individuals:")
    for name in registered_faces.keys():
        st.write(f"- {name}")

    run = st.checkbox('Start Live Recognition')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        name = recognize_face(frame, registered_faces)
        
     
        cv2.putText(frame, name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        FRAME_WINDOW.image(frame)

    camera.release()

# Streamlit app
def main():
    st.title("Face Recognition Attendance System")

    
    page = st.sidebar.selectbox("Choose a page", ["Register", "Live Recognition"])

    if page == "Register":
        st.header("Register a New Face")
        name = st.text_input("Enter the name of the person")
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if st.button("Register") and name and video_file:
            success = register_face(name, video_file)
            if success:
                st.success(f"Face registered for {name}")
            else:
                st.error("No face detected in the video. Please try again.")

    elif page == "Live Recognition":
        live_recognition()

if __name__ == "__main__":
    main()
