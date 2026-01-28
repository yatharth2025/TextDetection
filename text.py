import cv2
import pytesseract
import numpy as np
import pyttsx3 # <-- NEW: Import Text-to-Speech library

# --- Configuration (IMPORTANT: Update this path if on Windows) ---
# Set the path to the Tesseract executable
# Example for Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# If Tesseract is in your system PATH, you can set this to an empty string: r''
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the minimum confidence level for displaying text
CONFIDENCE_THRESHOLD = 65 

# --- Initialization ---
# Initialize the Text-to-Speech engine
tts_engine = pyttsx3.init() 

# --- Functions ---

def process_frame_for_text(frame, threshold):
    """Processes a single frame to detect and highlight text."""
    
    # Clone the frame so we don't modify the original snapshot
    processed_frame = frame.copy()
    
    # Convert to grayscale and then apply a slight blur to help OCR
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to get detailed box and text data
    data = pytesseract.image_to_data(gray_frame, output_type=pytesseract.Output.DICT)
    
    detected_words = []

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip() # Clean up the text
        conf = int(data['conf'][i])
        
        # Check if confidence is high enough AND the text is not empty/just spaces
        if conf > threshold and len(text) > 0 and text.isalnum(): # Added alnum check for cleaner results
            
            # Get bounding box coordinates (still needed for data even if not drawn)
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            
            # --- Draw on the Video Feed ---
            
            # 1. REMOVED: Drawing a green rectangle (bounding box)
            # cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # The rest of the drawing code for text was already commented out
            # ...

            # Store the detected word for console printing
            detected_words.append(text)

    return processed_frame, detected_words

# --- Main Script ---

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- Real-time Text Detector Initialized ---")
print("Press [SPACEBAR] to capture a frame and start text detection.")
print("Press [q] to quit the application.")

# State variable to hold the captured frame for persistent display
captured_image = None
# New state variable to hold the final transcribed text
transcribed_text = "N/A" 

while True:
    # Read a new frame from the webcam (always runs)
    ret, live_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Determine which image to display
    display_frame = live_frame.copy()
    
    # Get frame dimensions for placing the transcription
    (frame_h, frame_w) = display_frame.shape[:2]
    
    if captured_image is not None:
        # If an image was captured, display the processed result instead of the live feed
        # Note: The image is now just the captured photo, as no boxes were drawn on it
        display_frame = captured_image 
        
        # Add a text overlay to indicate detection is complete
        cv2.putText(display_frame, "SCAN COMPLETE - Press SPACE to SCAN new frame", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # The transcription overlay at the bottom was also previously removed.
        
    else:
        # Add a text overlay in live mode
        cv2.putText(display_frame, "LIVE FEED - Press SPACE to SCAN", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    # Show the current frame (live or processed)
    cv2.imshow('Text Detector', display_frame)
    
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    
    # 1. Quit condition ('q')
    if key == ord('q'):
        break
    
    # 2. Capture and Detect condition (Spacebar)
    elif key == ord(' '):
        print("\n--- CAPTURING AND SCANNING FRAME ---")
        
        # Process the newly captured live frame
        processed_img, detected_text_list = process_frame_for_text(live_frame, CONFIDENCE_THRESHOLD)
        
        # Update the captured_image state to the new processed image
        # Since no boxes are drawn, captured_image is now just the frozen photo.
        captured_image = processed_img 
        
        # Join the detected words into a single string for transcription
        if detected_text_list:
            transcribed_text = ' '.join(detected_text_list)
            print(f"✅ Text Detected: {transcribed_text}")
            
            # Speak the detected text
            tts_engine.say(f"Detected text: {transcribed_text}")
            tts_engine.runAndWait()
            
        else:
            transcribed_text = "NO TEXT DETECTED"
            print("❌ No text detected with high confidence.")

            # Speak the failure message
            tts_engine.say("No text detected with high confidence.")
            tts_engine.runAndWait()

# Cleanup
print("--- Text Detector Shutting Down ---")
cap.release()
cv2.destroyAllWindows()
