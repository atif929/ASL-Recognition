"""
================================================================================
IMPROVED ASL RECOGNITION APP - WITH TEXT BUILDING
================================================================================

New Features:
✨ Builds words from recognized letters (H-E-L-L-O → HELLO)
✨ Add space between words
✨ Delete last letter
✨ Clear all text
✨ Better prediction accuracy
✨ Smooth letter detection
================================================================================
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* NEW: Text display box */
    .text-display {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        min-height: 100px;
    }
    
    .text-content {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 3px;
        font-family: 'Courier New', monospace;
        word-wrap: break-word;
    }
    
    .text-label {
        font-size: 1rem;
        color: #bdc3c7;
        margin-bottom: 10px;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        margin: 20px 0;
    }
    
    .prediction-letter {
        font-size: 6rem;
        font-weight: 900;
        margin: 20px 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    .confidence-text {
        font-size: 1.8rem;
        margin-top: 15px;
        font-weight: 600;
    }
    
    .stat-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        text-align: center;
        margin: 15px 0;
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 1.1rem;
        color: #666;
        margin-top: 8px;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    
    .info-box h3, .info-box h4 {
        color: #ffffff !important;
        margin-top: 0;
        font-weight: 700;
    }
    
    .info-box p, .info-box ul, .info-box li {
        color: #e3f2fd !important;
        line-height: 1.6;
    }
    
    .info-box strong {
        color: #90caf9 !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #66bb6a;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    
    .success-box strong {
        color: #a5d6a7 !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #ff9800;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    
    .class-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_asl_model():
    """Load trained model and labels."""
    try:
        model = load_model('model/asl_model.h5')
        labels = np.load('model/labels.npy', allow_pickle=True)
        return model, labels
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

model, class_names = load_asl_model()

# =============================================================================
# SESSION STATE - WITH TEXT BUILDING
# =============================================================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=50)

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

if 'confident_predictions' not in st.session_state:
    st.session_state.confident_predictions = 0

if 'current_letter' not in st.session_state:
    st.session_state.current_letter = None

if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0

if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# NEW: Text building
if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = ""

if 'last_added_letter' not in st.session_state:
    st.session_state.last_added_letter = None

if 'last_add_time' not in st.session_state:
    st.session_state.last_add_time = 0

# =============================================================================
# PREPROCESSING (MUST MATCH TRAINING)
# =============================================================================

def preprocess_for_prediction(image, size=64):
    """Preprocess webcam frame - MUST MATCH train_improved.py"""
    
    # Convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization (IMPORTANT: matches training)
    img = cv2.equalizeHist(img)
    
    # Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Resize
    img = cv2.resize(img, (size, size))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add dimensions
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_sign(image, model, class_names, threshold=75.0):
    """Predict ASL sign from image."""
    
    if image is None or image.size == 0:
        return None, 0.0
    
    try:
        processed = preprocess_for_prediction(image, size=64)
        predictions = model.predict(processed, verbose=0)
        
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        
        if confidence >= threshold:
            predicted_letter = class_names[class_idx]
            return predicted_letter, confidence
        else:
            return None, confidence
            
    except Exception as e:
        return None, 0.0

# =============================================================================
# TEXT BUILDING FUNCTIONS
# =============================================================================

def add_letter_to_text(letter):
    """
    Add letter to recognized text with smart timing.
    Prevents duplicate letters within 2 seconds.
    """
    current_time = time.time()
    
    # Special handling for 'space' and 'del'
    if letter == 'space':
        if st.session_state.recognized_text and not st.session_state.recognized_text.endswith(' '):
            st.session_state.recognized_text += ' '
        st.session_state.last_add_time = current_time
        return True
    
    elif letter == 'del':
        if st.session_state.recognized_text:
            st.session_state.recognized_text = st.session_state.recognized_text[:-1]
        st.session_state.last_add_time = current_time
        return True
    
    elif letter == 'nothing':
        # Do nothing for 'nothing' class
        return False
    
    else:
        # Regular letter (A-Z)
        # Only add if enough time has passed since last addition
        if (st.session_state.last_added_letter != letter or 
            current_time - st.session_state.last_add_time > 2.0):
            
            st.session_state.recognized_text += letter
            st.session_state.last_added_letter = letter
            st.session_state.last_add_time = current_time
            return True
    
    return False


def clear_text():
    """Clear all recognized text."""
    st.session_state.recognized_text = ""
    st.session_state.last_added_letter = None
    st.session_state.last_add_time = 0


def create_confidence_bar(confidence):
    """Create confidence bar."""
    if confidence >= 80:
        color = "#4caf50"
    elif confidence >= 60:
        color = "#ff9800"
    else:
        color = "#f44336"
    
    return f"""
    <div style="background: #f0f0f0; border-radius: 15px; height: 35px; overflow: hidden; margin: 10px 0;">
        <div style="background: {color}; width: {confidence}%; height: 100%; 
                    display: flex; align-items: center; justify-content: center; 
                    color: white; font-weight: bold; font-size: 1.1rem;">
            {confidence:.1f}%
        </div>
    </div>
    """


def draw_roi_on_frame(frame, roi_coords, color=(0, 255, 0), thickness=3):
    """Draw ROI box on frame."""
    x, y, w, h = roi_coords
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(frame, "Position hand here", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-title">🤟 ASL Alphabet Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time Sign Language Recognition with Text Building</div>', unsafe_allow_html=True)
    
    if model is None or class_names is None:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Model Not Found</h3>
            <p>Please train the improved model first:</p>
            <code>python train_improved.py</code>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Model Loaded Successfully!</strong><br>
        Ready to recognize {len(class_names)} signs and build text!
    </div>
    """, unsafe_allow_html=True)
    
    # =============================================================================
    # SIDEBAR
    # =============================================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")
        
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold (%)",
            min_value=50,
            max_value=95,
            value=75,
            step=5
        )
        
        buffer_size = st.slider(
            "📊 Prediction Smoothing",
            min_value=3,
            max_value=15,
            value=7,
            step=1
        )
        
        st.markdown("---")
        st.markdown("## 📖 How to Use")
        st.markdown("""
        **Build Words:**
        1. Start camera
        2. Show letter signs (A-Z)
        3. Hold for 2 seconds
        4. Letter appears in text box!
        5. Show 'space' for spaces
        6. Show 'del' to delete
        
        **💡 Tips:**
        - Clear, distinct signs
        - Hold steady 2 seconds
        - Good lighting
        - Plain background
        """)
        
        st.markdown("---")
        st.markdown("## 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", st.session_state.total_predictions)
        with col2:
            st.metric("Confident", st.session_state.confident_predictions)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history.clear()
            st.session_state.total_predictions = 0
            st.session_state.confident_predictions = 0
            clear_text()
            st.success("History cleared!")
            st.rerun()
    
    # =============================================================================
    # TEXT DISPLAY (NEW FEATURE)
    # =============================================================================
    
    st.markdown("### 📝 Recognized Text")
    
    # Text display box
    display_text = st.session_state.recognized_text if st.session_state.recognized_text else "Text will appear here..."
    
    st.markdown(f"""
    <div class="text-display">
        <div class="text-label">Building Your Message:</div>
        <div class="text-content">{display_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("🔤 Add Space", use_container_width=True):
            if st.session_state.recognized_text and not st.session_state.recognized_text.endswith(' '):
                st.session_state.recognized_text += ' '
                st.rerun()
    
    with col_b:
        if st.button("⌫ Delete Last", use_container_width=True):
            if st.session_state.recognized_text:
                st.session_state.recognized_text = st.session_state.recognized_text[:-1]
                st.rerun()
    
    with col_c:
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_text()
            st.rerun()
    
    st.markdown("---")
    
    # =============================================================================
    # WEBCAM AND PREDICTION
    # =============================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Webcam Feed")
        
        webcam_placeholder = st.empty()
        
        if st.button("🎥 Start Camera" if not st.session_state.webcam_active else "⏹️ Stop Camera",
                     type="primary", use_container_width=True):
            st.session_state.webcam_active = not st.session_state.webcam_active
            st.rerun()
        
        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam")
                st.session_state.webcam_active = False
                return
            
            st.info("✅ Camera active! Show ASL signs to build text.")
            
            prediction_buffer = deque(maxlen=buffer_size)
            
            try:
                frame_count = 0
                
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    
                    roi_size = min(h, w) // 2
                    roi_x = (w - roi_size) // 2
                    roi_y = (h - roi_size) // 2
                    
                    roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
                    
                    if roi.size > 0:
                        predicted_letter, confidence = predict_sign(
                            roi, model, class_names, confidence_threshold
                        )
                        
                        st.session_state.total_predictions += 1
                        
                        if predicted_letter:
                            prediction_buffer.append(predicted_letter)
                            
                            if len(prediction_buffer) >= 5:
                                final_prediction = max(set(prediction_buffer), 
                                                     key=prediction_buffer.count)
                            else:
                                final_prediction = predicted_letter
                            
                            st.session_state.current_letter = final_prediction
                            st.session_state.current_confidence = confidence
                            st.session_state.confident_predictions += 1
                            
                            # Add to history
                            st.session_state.prediction_history.append({
                                'letter': final_prediction,
                                'confidence': confidence,
                                'time': time.strftime("%H:%M:%S")
                            })
                            
                            # Auto-add to text (NEW)
                            if frame_count % 30 == 0:  # Every 30 frames (~1 second)
                                add_letter_to_text(final_prediction)
                            
                            # Display on frame
                            cv2.putText(frame, f"{final_prediction}", 
                                      (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                                      2.5, (0, 255, 0), 5)
                            cv2.putText(frame, f"{confidence:.1f}%", 
                                      (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1.2, (0, 255, 0), 3)
                        else:
                            cv2.putText(frame, "Low Confidence", 
                                      (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 165, 255), 2)
                            st.session_state.current_letter = None
                    
                    frame = draw_roi_on_frame(frame, (roi_x, roi_y, roi_size, roi_size))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    time.sleep(0.03)
                    frame_count += 1
                    
                    if not st.session_state.webcam_active:
                        break
                        
            finally:
                cap.release()
                st.session_state.webcam_active = False
        
        else:
            placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Camera Inactive", 
                       (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (255, 255, 255), 3)
            webcam_placeholder.image(placeholder_img, channels="RGB", use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Current Prediction")
        
        if st.session_state.current_letter and st.session_state.current_confidence >= confidence_threshold:
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.5rem;">Detected Sign</div>
                <div class="prediction-letter">{st.session_state.current_letter}</div>
                <div class="confidence-text">{st.session_state.current_confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(create_confidence_bar(st.session_state.current_confidence), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>📌 Waiting...</strong><br>
                Make an ASL sign to see prediction!
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📝 Recent Predictions")
        
        if st.session_state.prediction_history:
            history_list = list(st.session_state.prediction_history)[-10:]
            history_list.reverse()
            
            for pred in history_list:
                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    st.markdown(f"**{pred['letter']}**")
                with col_b:
                    st.caption(pred['time'])
                with col_c:
                    st.caption(f"{pred['confidence']:.0f}%")
        else:
            st.info("History will appear here")
    
    # =============================================================================
    # REFERENCE
    # =============================================================================
    
    st.markdown("---")
    st.markdown("## 🔤 ASL Alphabet Reference")
    
    st.markdown("""
    <div class="info-box">
        <h4>📚 Special Signs</h4>
        <ul>
            <li><strong>A-Z:</strong> Standard alphabet letters</li>
            <li><strong>space:</strong> Add space between words</li>
            <li><strong>del:</strong> Delete last character</li>
            <li><strong>nothing:</strong> No action (neutral)</li>
        </ul>
        <p>💡 Hold each sign steady for 2 seconds to add to text!</p>
    </div>
    """, unsafe_allow_html=True)
    
    badges_html = '<div style="text-align: center; margin: 20px 0;">'
    for class_name in class_names:
        badges_html += f'<span class="class-badge">{class_name}</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 30px;">
        <p style="font-size: 1.2rem;">🤟 <strong>ASL Recognition with Text Building</strong></p>
        <p>Spell words letter by letter • Build sentences • Communicate freely</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()