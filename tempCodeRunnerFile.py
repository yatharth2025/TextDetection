import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


CONFIDENCE_THRESHOLD = 65 



def process_frame_for_text(frame, threshold):
    """Processes a single frame to detect and highlight text."""
    
    processed_frame = frame.copy()
    
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    
    data = pytesseract.image_to_data(gray_frame, output_type=pytesseract.Output.DICT)
    
    detected_words = []

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip() 
        conf = int(data['conf'][i])
        
        if conf > threshold and len(text) > 0 and text.isalnum():
            
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            detected_words.append(text)

    return processed_frame, detected_words
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- Real-time Text Detector Initialized ---")
print("Press [SPACEBAR] to capture a frame and start text detection.")
print("Press [q] to quit the application.")
captured_image = None

while True:
    
    ret, live_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

   
    display_frame = live_frame.copy()
    
    if captured_image is not None:
       
        display_frame = captured_image 
        
        
        cv2.putText(display_frame, "SCAN COMPLETE - Press SPACE to SCAN new frame", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display_frame, "LIVE FEED - Press SPACE to SCAN", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Text Detector', display_frame)

    key = cv2.waitKey(1) & 0xFF
   
    if key == ord('q'):
        break

    elif key == ord(' '):
        print("\n--- CAPTURING AND SCANNING FRAME ---")
        
      
        processed_img, detected_text_list = process_frame_for_text(live_frame, CONFIDENCE_THRESHOLD)
        
       
        captured_image = processed_img
        
        if detected_text_list:
            print(f"✅ Text Detected: {' '.join(detected_text_list)}")
        else:
            print("❌ No text detected with high confidence.")


print("Text Detector Shutting Down")
cap.release()
cv2.destroyAllWindows()
