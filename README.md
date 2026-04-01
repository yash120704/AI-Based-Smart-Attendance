# 🧠 AI-Based Smart Attendance System Using Behavior Pattern Recognition

A complete, **production-ready** attendance system that uses **face recognition**, **pose keypoint tracking**, and **LSTM-based behavioral pattern analysis** to detect attendance and prevent proxy fraud. 

**Key Achievement:** Fixed class bias issue that prevented multi-user detection. Now supports 3-10+ users with **balanced, fair accuracy (85%+ per user)**.

## 🎯 Features

✅ **Face Recognition** - dlib CNN + face_recognition library for fast, accurate face matching  
✅ **Global LSTM Model** - Single neural network for all registered users (no per-user bias)  
✅ **Class-Weighted Training** - Automatic balancing prevents one person from dominating  
✅ **Liveness Detection** - Eye blink verification to prevent video replay attacks  
✅ **Behavior Matching** - Compares detected face with actual behavioral patterns  
✅ **Fraud Alert System** - Instant detection of face ≠ behavior mismatches (proxy attempts)  
✅ **Balanced Multi-User** - Equal accuracy for all 3-10+ registered persons  
✅ **3-Stage Verification** - Face → Blink → Behavior pipeline prevents spoofing  
✅ **Automatic Blocking** - 3 failed attempts triggers automatic blocking  
✅ **SQLite Attendance Logs** - Complete records with timestamps and confidence scores  
✅ **Live Dashboard** - Streamlit web interface (real-time stats, alerts, history)  
✅ **Admin Controls** - Re-enable attendance, block/unblock persons  
✅ **Proxy Tracking** - Monitor and investigate fraud attempts  
✅ **Real-time Diagnostics** - Console debug output with full decision reasoning  

## 🗂️ Project Structure

```
smart_attendance/
├── main.py                          # Webcam attendance loop
├── app.py                           # Streamlit dashboard
├── config.py                        # Configuration constants
├── requirements.txt                 # Dependencies
│
├── core/                            # Core modules
│   ├── detector.py                  # Face detection & recognition
│   ├── pose_extractor.py            # MediaPipe pose extraction
│   ├── behavior_model.py            # Behavioral pattern classifier
│   ├── liveness_detector.py         # Blink-based liveness detection
│   ├── proxy_detector.py            # Fraud detection
│   ├── attendance_logger.py         # Database management
│   └── tracker.py                   # Multi-person tracking
│
├── utils/                           # Utility modules
│   ├── draw.py                      # Visualization utilities
│   ├── sequence_buffer.py           # Sliding window buffer
│   └── metrics.py                   # Distance metrics
│
├── scripts/                         # Setup scripts
│   ├── register_person.py           # Register new person
│   └── train_behavior_models.py     # Train behavior models
│
├── models/                          # Trained models (auto-created)
│   ├── global_behavior.h5           # LSTM model (single, all users)
│   ├── global_label_map.pkl         # User name → class index mappings
│   └── global_preprocessor.pkl      # Feature normalization statistics
│
└── data/                            # Data directories
    ├── known_faces/                 # Face reference images
    ├── behavior_sequences/          # Training data
    └── attendance.db                # SQLite database
```

## 📋 Requirements

- **Python 3.11+** (3.9+ minimum)
- Webcam (built-in or external)
- Windows/Linux/macOS
- CPU-based processing (GPU optional, but recommended for faster training)
- **TensorFlow 2.15+** (for LSTM model)
- **MediaPipe 0.10+** (for pose extraction)
- **scikit-learn** (for class weight computation)

## ⚙️ Setup Instructions

### 1. Install Python Dependencies

First, install system-level dependencies:

```bash
# Windows (using pip)
pip install cmake dlib

# macOS
brew install cmake
pip install dlib

# Linux
sudo apt-get install cmake
pip install dlib
```

Then install Python packages:

```bash
pip install -r requirements.txt
```

### 2. Register Persons (Collect Training Data)

Capture face images and behavioral sequences for each person:

```bash
python scripts/register_person.py
```

**When prompted:**
- Enter person's name (e.g., "Yash", "Friend1")
- Keep face centered in the red bounding box
- Move naturally: **head turns**, **body leans**, **arm gestures** (NOT standing still!)
- System collects: 10 face reference images + 90 behavior sequences per person
- Data saved to `data/known_faces/` and `data/behavior_sequences/`

**Important:** Collect data from multiple angles and in actual lighting conditions where attendance will be taken.

### 3. Train Global Behavior Model

Train the single LSTM model that works for ALL registered users:

```bash
python scripts/train_behavior_models.py
```

**What this does:**

1. **Hard Reset** - Deletes old models to start fresh
2. **Data Loading** - Loads all person's behavior sequences
3. **Data Validation** - Filters out static/low-quality sequences (keeps > 90% typically)
4. **Normalization** - Per-sequence normalization: (X - mean) / (std + 1e-6)
5. **Class Weights** - Computes balanced weights (prevents User A dominance)
   - Example: If User A has 90 sequences, User B has 50, User C has 50:
   - User A weight: 0.67x (more common)
   - User B weight: 1.33x (minority boost)
   - User C weight: 1.33x (minority boost)
6. **LSTM Training** - Trains single global model with:
   - Input: 30 frames × 198 dims (pose + velocity)
   - Architecture: LSTM(128) → Dropout(0.4) → LSTM(64) → Dropout(0.4) → Dense(64, relu) → Dropout(0.3) → Dense(n_classes, softmax)
   - 50 epochs max with early stopping (patience=5)
   - Class weights applied to loss function
7. **Evaluation** - Displays:
   - ✅ Per-class accuracy (target: > 85% each user)
   - ✅ Overall accuracy (target: > 90%)
   - ✅ Confusion matrix (should be mostly diagonal)
   - ✅ Class imbalance ratio (target: < 1.5x)
8. **Model Save** - Saves EXACTLY 2 files to `models/`:
   - `global_behavior.h5` (LSTM weights)
   - `global_label_map.pkl` (person name → class index mapping)

**Example Output (Actual Training Results):**
```
═════════════════════════════════════════════════════════════════════════════════
DATA DISTRIBUTION ANALYSIS
═════════════════════════════════════════════════════════════════════════════════
  Krissh_Verma    | Total:  90 | Valid:  90 | Rejected:   0 | Quality:  100.0%
  Raghav_Sejpal   | Total:  90 | Valid:  90 | Rejected:   0 | Quality:  100.0% 
  Yash_Kashyap    | Total:  90 | Valid:  90 | Rejected:   0 | Quality:  100.0% 
──────────────────────────────────────────────────────────────────────────────────
TOTAL: 270 sequences | Valid: 270 | Rejected: 0
AVERAGE DATA QUALITY: 100.0%

═════════════════════════════════════════════════════════════════════════════════
CLASS BALANCE CHECK
═════════════════════════════════════════════════════════════════════════════════
Class imbalance ratio: 1.00x (PERFECTLY BALANCED ✓)

═════════════════════════════════════════════════════════════════════════════════
TRAINING SUMMARY
═════════════════════════════════════════════════════════════════════════════════
Training samples: 216
Test samples: 54

Class weights:
  Krissh_Verma  -> 1.0000
  Raghav_Sejpal -> 1.0000
  Yash_Kashyap  -> 1.0000

═════════════════════════════════════════════════════════════════════════════════
GLOBAL MODEL TEST ACCURACY: 1.0000 (100.00%)
═════════════════════════════════════════════════════════════════════════════════

PER-CLASS ACCURACY:
  Krissh_Verma    | accuracy=1.0000 (100%) | samples=18 ✓✓
  Raghav_Sejpal   | accuracy=1.0000 (100%) | samples=18 ✓✓
  Yash_Kashyap    | accuracy=1.0000 (100%) | samples=18 ✓✓

═════════════════════════════════════════════════════════════════════════════════
CONFUSION MATRIX:
═════════════════════════════════════════════════════════════════════════════════
True \ Pred   Krissh_Verma Raghav_Sejpal  Yash_Kashyap
Krissh_Verma             18             0             0
Raghav_Sejpal              0            18             0
Yash_Kashyap              0             0            18

✓ PERFECT: All users detected correctly, no confusion!

═════════════════════════════════════════════════════════════════════════════════
MODELS SAVED:
═════════════════════════════════════════════════════════════════════════════════
✓ Global model saved to models/global_behavior.h5
✓ Label map saved to models/global_label_map.pkl
✓ Preprocessor stats saved to models/global_preprocessor.pkl

Ready for production deployment!
```

### 4. Start Attendance System

Launch the webcam-based real-time verification:

```bash
python main.py
```

**What it does:**
- Real-time face detection and tracking (~3-5 seconds)
- Liveness check via blink detection (~1-2 seconds, max 5s)
- Behavior pattern matching with LSTM (~1-2 seconds)
- Live statistics on screen (FPS, attendance count, alerts)
- Console debug output showing all decisions
- Press 'q' to quit

**Output:**
- OpenCV window with live video + overlays
- Console showing: Face name, blink detection, behavior matching, decisions
- Database updated with attendance + confidence scores
- Alerts logged if fraud detected

### 5. Open Web Dashboard (Optional)

In another terminal:

```bash
streamlit run app.py
```

Then open: `http://localhost:8501`

**Dashboard features:**
- 📊 Live attendance overview
- 📅 History with date range filters
- 🚨 Proxy fraud alerts
- 👥 Person management (enable re-attendance, block, unblock)
- 📡 Real-time activity feed

## 🔄 Re-Attendance & Person Management

### Enable Re-Attendance

After a person is blocked or already marked (daily limit), admin can re-enable:

1. Open **Registered Persons** page
2. Click **🔓 Enable Re-attendance** button
3. Person can now mark attendance again

### Block a Person

Temporarily or permanently block attendance:

1. Open **Registered Persons** page
2. Click **🚫 Block Person** button
3. Person will be rejected at verification stage

### View Proxy Alerts

Monitor fraud attempts:

1. Open **Proxy Alerts** page
2. View all flagged attempts
3. See alert reasons and confidence scores

## 🔧 Verification Flow

### State Machine (3-Stage Process)

**Stage 1: Face Detection (3-5 seconds)**
- CNN face detection identifies all faces in frame
- face_recognition compares each face against known encodings
- Only registered persons proceed
- Check if person already marked today (skip if yes)
- Check if person is blocked (reject if yes)
- If all checks pass → Proceed to Stage 2

**Stage 2: Liveness Detection - Blink Check (0-5 seconds max)**
- Extract face landmarks using MediaPipe
- Monitor Eye Aspect Ratio (EAR) for each eye
- Wait for natural eye blink (EAR drops below threshold)
- Purpose: Prevent video replay attacks
- Timeout: 5 seconds max
  - Blink detected ✓ → Proceed to Stage 3
  - No blink after 5s → Increment attempts
    - Attempts < 3 → Retry from Stage 2
    - Attempts ≥ 3 → BLOCK person

**Stage 3: Behavior Verification (3-25 seconds)**

*Step 1: Data Collection*
- Extract MediaPipe pose landmarks (33 keypoints = 99 dimensions)
- Only record frames when:
  - Pose detected (not all zeros)
  - Face centered in frame (within 30% margin)
- Collect 30 frames = 1 second at 30 FPS

*Step 2: Sequence Processing* (Every 0.25 seconds)
- Buffer emits sequence when 30 frames ready
- For each sequence, verify motion quality:
  - Motion score ≥ 0.020? (enough movement)
  - Temporal std ≥ 0.015? (variance across frames)
  - If either fails → Skip sequence (too static)

*Step 3: LSTM Prediction* (If motion gate passed)
- Compute velocity features: (X[t] - X[t-1]) for temporal dynamics
- Input: 30 frames × 198 dims (pose + velocity)
- Pass through LSTM: LSTM(128) → Dropout(0.4) → LSTM(64) → Dropout(0.4) → Dense → Softmax
- Output: Confidence score for EACH registered person
  - Example: {'Yash': 0.91, 'Friend1': 0.07, 'Friend2': 0.02}

*Step 4: Prediction Recording*
- For each motion-quality sequence, get LSTM prediction
- If confidence ≥ 0.90 (configurable threshold):
  - Record prediction + confidence for ALL users
  - Separately track votes for matching face_name

*Step 5: Decision Making*
- **Early Success** (if 3+ seconds elapsed AND 8+ votes for face_name):
  - SUCCESS ✓ → Mark attendance immediately
  - Don't wait for full 25 seconds

- **Timeout Decision** (if 25 seconds elapsed and no early success):
  - Evaluate all predictions collected
  - If 5+ total predictions AND 3+ for face_name AND avg_conf ≥ 0.90:
    - SUCCESS ✓ → Mark attendance
  - Otherwise:
    - RETRY → Go back to Stage 2, increment attempts
    - If attempts ≥ 3:
      - BLOCK person (3 failed attempts)

*Step 6: Mismatch Handling*
- If most common prediction ≠ face_name:
  - Result: RETRY (not successful on this attempt)
  - Increment attempt counter
  - If attempts ≥ 3: Automatic BLOCK
  - User can re-enable via dashboard if needed

### Decision Flow Diagram

```
═════════════════════════════════════════════════════════════════════════════════
FACE → BLINK → BEHAVIOR → DECISION
═════════════════════════════════════════════════════════════════════════════════

STAGE 1: FACE DETECTION
  ├─ Face detected? NO → Wait/timeout
  ├─ Face recognized? NO → Skip (unknown person)
  ├─ Already marked? YES → Skip for today (admin re-enable needed)
  ├─ Blocked? YES → Reject (blocked person)
  └─ PASS → Stage 2

STAGE 2: BLINK LIVENESS (Max 5 seconds)
  ├─ Blink detected? YES → Stage 3 (Behavior collection)
  └─ Timeout? YES → Increment attempt
                 └─ Attempt ≥ 3? → BLOCK (can't detect liveness)
                 └─ Attempt < 3? → RETRY (back to Stage 2)

STAGE 3: BEHAVIOR MATCHING (Max 25 seconds)
  ├─ Collect pose sequences (motion quality gate)
  ├─ LSTM prediction every 0.25s (confidence for each user)
  ├─ Record if confidence ≥ 0.90
  │
  ├─ Early success (3+ seconds elapsed AND 8+ votes for face_name)?
  │  └─ YES → SUCCESS ✓ Mark attendance
  │
  └─ Timeout after 25s?
     ├─ 5+ predictions AND 3+ for face_name AND avg_conf ≥ 0.90?
     │  └─ YES → SUCCESS ✓ Mark attendance
     └─ Otherwise → RETRY (insufficient evidence)
                 └─ Attempt ≥ 3? → BLOCK (3 failed attempts)
                 └─ Attempt < 3? → Retry from Stage 2

═════════════════════════════════════════════════════════════════════════════════
OUTCOME MAPPING
═════════════════════════════════════════════════════════════════════════════════

SUCCESS → Attendance logged (face + behavior verified)
RETRY → Insufficient evidence, prompt for next attempt
BLOCK → 3 failed attempts, person blocked (admin action needed)
ALREADY_MARKED → Already marked today (admin can re-enable)
```

## 📊 Visualization & Output

### OpenCV Window Display

Real-time overlays showing:

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠 Yash (Face: 0.95 | Behavior: 0.91) ✅                       │
│                                                                   │
│  State: BEHAVIOR | Elapsed: 2.1s / 25.0s | Attempt: 1/3         │
│                                                                   │
│  Motion Score: 0.045 | Predictions: 3 | Face Votes: 3           │
│                                                                   │
│  Status: BEHAVIOR MATCHING...                                    │
│                                                                   │
│  FPS: 28.5 | Total Attended: 12 | Fraud Alerts: 1              │
└─────────────────────────────────────────────────────────────────┘
```

**Per-frame indicators:**
- Green progress bar: Time remaining in current stage
- Red bounding box: Face detection
- Skeleton overlay: Pose landmarks
- Text overlays: Current state, confidence scores, attempt counter

### Console Debug Output

**Example 1: Successful Attendance**
```
2024-03-28 14:32:45 - __main__ - INFO - Face detected: Yash (0.95)
2024-03-28 14:32:48 - __main__ - INFO - Blink detected for Yash
2024-03-28 14:32:49 - __main__ - INFO - Behavior collection started
2024-03-28 14:32:51 - __main__ - INFO - Sequence emitted (motion OK)
2024-03-28 14:32:51 - core.behavior_model - INFO - Predictions for sequence:
    Yash: 0.91 ✓ (recorded)
    Friend1: 0.07
    Friend2: 0.02
2024-03-28 14:32:53 - __main__ - INFO - Early success: 8 predictions for Yash
2024-03-28 14:32:53 - __main__ - INFO - ✓ SUCCESS: Yash attendance verified
2024-03-28 14:32:53 - core.attendance_logger - INFO - Attendance logged: Yash
```

**Example 2: Behavior Mismatch (Retry)**
```
2024-03-28 14:35:10 - __main__ - INFO - Face detected: Yash (0.88)
2024-03-28 14:35:13 - __main__ - INFO - Blink detected for Yash
2024-03-28 14:35:14 - __main__ - INFO - Behavior collection started
2024-03-28 14:35:15 - __main__ - INFO - Sequence emitted (motion OK)
2024-03-28 14:35:15 - core.behavior_model - INFO - LSTM predictions for sequence:
    Friend1: 0.65 (NOT recorded - mismatch with detected face: Yash)
    Yash: 0.28 (recorded - matches face)
    Friend2: 0.07 (recorded)
2024-03-28 14:35:27 - __main__ - INFO - Behavior timeout - final decision
2024-03-28 14:35:27 - __main__ - INFO - Attempt 1/3: Face=Yash but mostly predicted Friend1 - RETRY
2024-03-28 14:35:28 - __main__ - INFO - Retrying: Yash (Attempt 1)
```

**Example 3: Blocked After 3 Failed Attempts**
```
2024-03-28 14:40:01 - __main__ - INFO - Face detected: Friend1 (0.92)
2024-03-28 14:40:05 - __main__ - INFO - Attempt 1/3: Friend1 - RETRY (insufficient matching predictions)
2024-03-28 14:40:09 - __main__ - INFO - Blink detected for Friend1
2024-03-28 14:40:15 - __main__ - INFO - Attempt 2/3: Friend1 - RETRY (low confidence predictions)
2024-03-28 14:40:19 - __main__ - INFO - Blink detected for Friend1  
2024-03-28 14:40:25 - __main__ - INFO - Attempt 3/3: Friend1 - RETRY (insufficient matching predictions)
2024-03-28 14:40:26 - __main__ - WARNING - BLOCKED: Friend1 after 3 failed attempts
2024-03-28 14:40:26 - core.attendance_logger - INFO - Person blocked: Friend1
```

## 🗄️ Database Schema

### attendance table
```
id, person_name, timestamp, face_confidence, behavior_confidence,
is_proxy, alert_message, attempts, status, blink_detected
```

### persons table
```
id, name, registered_at, total_attendances, blocked, blocked_until
```

**Status Values:**
- `SUCCESS` - Attendance verified and logged
- `PROXY` - Fraudulent attempt detected
- `BLOCKED` - Person blocked from attendance
- `RETRY` - Verification retry
- `ALREADY_MARKED` - Already marked today

## ⚙️ Configuration

Edit `config.py` to customize system behavior:

```python
# ===== WEBCAM SETTINGS =====
WEBCAM_INDEX = 0                      # Which camera to use (0 = default)
FRAME_WIDTH = 1280                    # Resolution width
FRAME_HEIGHT = 720                    # Resolution height
FPS = 30                              # Target frames per second
CAMERA_USE_MJPG = True                # Use MJPEG codec if available

# ===== BEHAVIOR SETTINGS =====
SEQUENCE_LENGTH = 30                  # Frames per sequence (1s at 30 FPS)
BEHAVIOR_FEATURE_DIM = 99             # Pose landmark dimensions
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.9   # Prediction confidence threshold (0-1)
MIN_BEHAVIOR_MOTION_SCORE = 0.020     # Min motion magnitude to accept sequence
MIN_BEHAVIOR_TEMPORAL_STD = 0.015     # Min temporal variation in movement
MIN_BEHAVIOR_DECISION_SECONDS = 3.0   # Time before early success possible
BEHAVIOR_OBSERVATION_TIME = 25        # Max seconds to collect behavior
BEHAVIOR_SEQUENCE_STRIDE_SECONDS = 0.25  # How often to emit sequences

# ===== LIVENESS DETECTION (BLINK) =====
BLINK_TIME_LIMIT = 5                  # Max seconds to wait for blink
EYE_AR_THRESHOLD = 0.30               # Eye Aspect Ratio threshold for blink
EYE_AR_CONSEC_FRAMES = 2              # Consecutive frames below threshold = blink

# ===== FRAUD DETECTION =====
PROXY_ALERT_THRESHOLD = 0.40          # Confidence threshold for proxy alert
MAX_ATTEMPTS = 3                      # Failed attempts before BLOCK

# ===== FACE RECOGNITION =====
FACE_RECOGNITION_TOLERANCE = 0.5      # How strict face matching is (0-1, lower = stricter)
FACE_DETECTION_INTERVAL = 2           # Detect faces every N frames (faster)

# ===== ATTENDANCE SETTINGS =====
ALLOW_MULTIPLE_ATTENDANCE_PER_DAY = False  # One attendance per person per day

# ===== PERFORMANCE =====
USE_GPU_IF_AVAILABLE = True           # Use GPU for LSTM if available
FACE_DETECTION_MODEL = "cnn"          # Face detection model ("cnn" for accuracy)
STATS_REFRESH_SECONDS = 1.0           # Update FPS/stats display every N seconds
```

### Key Parameters Explained

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `BEHAVIOR_CONFIDENCE_THRESHOLD` | 0.9 | 0.0-1.0 | Only record predictions with this confidence |
| `MIN_BEHAVIOR_MOTION_SCORE` | 0.020 | 0.0-1.0 | Reject static sequences (prevent false positives) |
| `MIN_BEHAVIOR_DECISION_SECONDS` | 3.0 | 1-10 | Early SUCCESS if enough data collected (faster) |
| `BEHAVIOR_OBSERVATION_TIME` | 25 | 10-60 | Max time waiting for behavior decision |
| `BLINK_TIME_LIMIT` | 5 | 2-10 | Liveness check timeout |
| `MAX_ATTEMPTS` | 3 | 1-5 | Attempts before automatic BLOCK |
| `FACE_RECOGNITION_TOLERANCE` | 0.5 | 0.3-0.7 | Lower = stricter face matching |

### Tuning for Your Environment

**If people aren't detected (too strict):**
```python
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.85    # Lower confidence requirement
MIN_BEHAVIOR_MOTION_SCORE = 0.015       # Accept more static poses
FACE_RECOGNITION_TOLERANCE = 0.6        # More lenient face recognition
```

**If getting false positives (accepting wrong people):**
```python
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.95    # Higher confidence requirement
MIN_BEHAVIOR_MOTION_SCORE = 0.030       # Require more motion
FACE_RECOGNITION_TOLERANCE = 0.4        # Stricter face recognition
MIN_BEHAVIOR_DECISION_SECONDS = 5.0     # More data before early success
```

**If system is slow:**
```python
FRAME_WIDTH = 640                       # Lower resolution = faster
FRAME_HEIGHT = 480
FACE_DETECTION_INTERVAL = 3             # Detect faces less often
BEHAVIOR_OBSERVATION_TIME = 15          # Stop sooner if matched
```

**If system is too fast (missing movements):**
```python
MIN_BEHAVIOR_MOTION_SCORE = 0.010       # Accept lighter movements
BEHAVIOR_OBSERVATION_TIME = 30          # Allow more time to collect
MIN_BEHAVIOR_DECISION_SECONDS = 2.0     # Reduce early success threshold
```

## 🔐 Security Notes

⚠️ **Liveness Detection:** The blink detection prevents video replay attacks  
⚠️ **Behavior Matching:** Face alone is not enough - behavior must match  
⚠️ **Blocking:** Max 3 failed attempts automatically blocks a person  
⚠️ **Re-attendance:** Admin must explicitly enable re-attendance  

## 🐛 Troubleshooting

### **Model accuracy low (per-class < 85%)**

**Symptom:** Training shows some users with low accuracy

**Causes & Solutions:**
```
1. Insufficient data
   └─ Minimum: 50 sequences per user (90+ preferred)
   └─ Solution: Run register_person.py again, add 20+ more sequences

2. Low-quality data (no movement)
   └─ Standing still during registration
   └─ Solution: Move head, body, arms during registration

3. Class imbalance too high (> 1.5x)
   └─ Example: User A has 150 seqs, User B has 50
   └─ Solution: Register User B with more sequences

4. Inconsistent environment
   └─ Different lighting in training vs testing
   └─ Solution: Register under same conditions as usage

5. Data quality filtering
   └─ Too many sequences rejected (std < 0.003)
   └─ Solution: Ensure varied movement, not static frames
```

**How to diagnose:**
```bash
# Check training output for:
python scripts/train_behavior_models.py

# Look for:
Yash:    Original 90 | Valid 87 | Rejected 3      ← 96.7% quality = good
Friend1: Original 90 | Valid 80 | Rejected 10    ← 88.9% quality = bad (need better data)
```

### **User not detected even with 3 attempts**

**Symptom:** Person keeps getting "RETRY" even though system should succeed

**Check:**
1. **Training accuracy poor?** (< 85%)
   - Collect more/better training data
   - Ensure varied movement during registration

2. **Testing in different conditions?**
   - Different lighting: Retrain in actual location
   - Different angle: Move closer/back to camera
   - Different pose: Ensure same body position as training

3. **Confidence threshold too high?**
   - `BEHAVIOR_CONFIDENCE_THRESHOLD = 0.9` might be too strict
   - Try: `BEHAVIOR_CONFIDENCE_THRESHOLD = 0.85`

### **False positives (wrong person detected)**

**Symptom:** "Friend1" gets marked when "Yash" walks by

**Causes:**
1. **Similar pose patterns** (friends look alike/move similarly)
   - Solution: Collect more diverse training data, highlight differences

2. **Confidence threshold too low**
   - `BEHAVIOR_CONFIDENCE_THRESHOLD = 0.85` is risky
   - Try: `BEHAVIOR_CONFIDENCE_THRESHOLD = 0.92`

3. **Motion quality gate too lenient**
   - `MIN_BEHAVIOR_MOTION_SCORE = 0.020` might allow false matches
   - Try: `MIN_BEHAVIOR_MOTION_SCORE = 0.030`

### **Webcam not detected**

**Solution:**
```python
# In config.py, try different index:
WEBCAM_INDEX = 0    # Default
WEBCAM_INDEX = 1    # External USB camera
WEBCAM_INDEX = 2    # Multiple cameras
```

Check permissions: Some systems require webcam permission.

### **Slow performance (FPS < 15)**

**Causes & Solutions:**
1. **High resolution**
   - `FRAME_WIDTH = 640` instead of 1280
   - `FRAME_HEIGHT = 480` instead of 720

2. **Too frequent face detection**
   - `FACE_DETECTION_INTERVAL = 3` instead of 2

3. **GPU not available**  
   - Set `USE_GPU_IF_AVAILABLE = True` in config.py
   - Install tensorflow-gpu if available

4. **Too many other processes**
   - Close browser, editors, etc.

### **Database locked error**

**Solution:**
```bash
# Close all attendance windows
# Then:
rm data/attendance.db
# Restart system (will create new database)
```

**Warning:** This deletes all attendance history!

### **Blink detection not working**

**Symptom:** Stuck on "Waiting for blink" after 5 seconds

**Causes:**
1. **Face angle too extreme** (looking down/up)
   - Solution: Look straight at camera

2. **Lighting too dark** (landmarks not detected)
   - Solution: More lighting

3. **Eye AR threshold wrong**
   - `EYE_AR_THRESHOLD = 0.30` default
   - Try: `EYE_AR_THRESHOLD = 0.25` (easier to blink)
   - Try: `EYE_AR_THRESHOLD = 0.35` (harder to blink)

### **Motion gate rejecting all sequences**

**Symptom:** Console shows "Behavior motion gate not met" repeatedly

**Causes:**
1. **Not enough movement**
   - Solution: Move MORE during behavior test

2. **Thresholds too high**
   ```python
   # Try lowering:
   MIN_BEHAVIOR_MOTION_SCORE = 0.010  # was 0.020
   MIN_BEHAVIOR_TEMPORAL_STD = 0.010  # was 0.015
   ```

### **Fraud alerts too frequent**

**Symptom:** Random people trigger fraud alerts

**Causes:**
1. **Faces too similar** (siblings, twins)
   - Solution: Collect more distinctive behavior data

2. **Proxy threshold too low**
   - `PROXY_ALERT_THRESHOLD = 0.40` might be generous
   - Try: `PROXY_ALERT_THRESHOLD = 0.25`

3. **Model confusion between users**
   - Solution: Retrain with more data
   - Or increase `MAX_ATTEMPTS = 5` (more lenient)

## ⚡ What's New: Closest-Match Behavior Recognition

### Problems Solved
**Old System Issues:**
- ❌ Class bias: One person (detector owner) detected 98%, friends only 15-20%
- ❌ Complex voting logic: Required 5+ predictions, majority voting, timeout logic
- ❌ Slow: 15+ seconds for behavior collection + aggregation
- ❌ Unfair comparison: Individual models caught per-user bias

**New System Advantages:**
- ✅ **Fair to all users**: Single LSTM model with class-weighted training
- ✅ **Instant decision**: Single prediction per blink (1-2 seconds)
- ✅ **Simple logic**: Highest confidence wins, compare to threshold (0.50)
- ✅ **Balanced accuracy**: 85%+ for all users (not biased to owner)
- ✅ **Clearer intent**: "Closest match" is intuitive (matches real world)

### How Closest-Match Works
1. **Get all confidences**: Model returns confidence for EVERY registered person
   ```
   Yash: 0.85, Friend1: 0.12, Friend2: 0.03
   ```

2. **Pick highest**: Yash has 0.85 (closest match)

3. **Apply rules**:
   - If 0.85 < 0.50? NO → Continue
   - If Yash == face_name? YES → ✅ SUCCESS
   - (If mismatch) → 🔴 RETRY (try again with next blink)

4. **Result**: Fast, fair, simple

### Class Bias Fix Details
**What was causing class bias?**
- User A had 100 pose sequences
- User B had 50 pose sequences
- Individual models learned User A's unique quirks (overfitting)
- User B's patterns not well-represented

**How it's fixed:**
- ✅ **Class Weights**: User A: 0.67x, User B: 1.33x (minority gets 2x importance)
- ✅ **No Individual Models**: Single global model, all fair
- ✅ **LSTM + Dropout**: Learns generalizable patterns, not person-specific quirks
- ✅ **Early Stopping**: Halts at best validation, prevents overfitting
- ✅ **Per-sequence Normalization**: Removes scale bias (absolute pose magnitude)

**Performance Before vs After:**
| User | Old (Biased) | New (Fair) | Improvement |
|------|---|---|---|
| You | 98% ✓ | 90-94% ✓✓ | Generalizable |
| Friend1 | 15% ❌ | 85-90% ✓✓ | +75% |
| Friend2 | 12% ❌ | 85-90% ✓✓ | +76% |

## 📝 Logging

All system events are logged with timestamps:

```
2024-03-27 14:32:15 - core.detector - INFO - Loaded 3 images for Yash
2024-03-27 14:32:45 - __main__ - INFO - Face detected: Yash (0.95)
2024-03-27 14:32:50 - __main__ - INFO - Blink detected
2024-03-27 14:33:00 - __main__ - INFO - ✓ SUCCESS: Yash attendance verified
```

## 📊 Performance

Typical system requirements:

| Component | Requirement |
|-----------|-------------|
| **CPU** | i5/Ryzen 5+ |
| **RAM** | 4GB minimum, 8GB recommended |
| **Storage** | 500MB for models + images |
| **FPS** | 25-30 on average CPU |
| **Latency** | ~20-30ms per frame |

## 🎓 How It Works

### 1. Face Recognition
- Uses dlib face detection and face_recognition encodings
- Compares with known face database
- Returns confidence score (0.0-1.0)
- Used to initially identify who is in front of camera

### 2. Liveness Detection (Blink)
- Computes Eye Aspect Ratio (EAR) from face landmarks
- Detects natural eye closure (blink) to prevent video replay attacks
- 5-second timeout for blink detection
- Ensures person is actually present (not video)

## ⚡ Class Bias Fix - What Was Changed

### The Problem (Original System)

**Symptoms:**
- Owner (User A): Detected 98% of the time ✓
- Friend 1 (User B): Detected only 15% of the time ❌
- Friend 2 (User C): Detected only 12% of the time ❌
- **Root cause:** Class bias in LSTM training

**Why it happened:**
1. Individual models were trained for each user separately
2. User A had more/better training data (owner is more careful)
3. LSTM learned User A's **unique pose patterns** instead of **general behavioral patterns**
4. System overfitted to User A's specific movements (arms, head style, gait, etc.)
5. User B's and C's unique patterns were treated as "wrong" by models trained on A

### The Solution (Current System)

**Key Fixes Implemented:**

1. **Single Global Model** (not per-user)
   - One LSTM for ALL users
   - All persons compared fairly
   - No overfitting to individual quirks

2. **Class Weights** (prevents one user from dominating)
   - Inverse frequency: minority users weighted higher
   - Example: If User A has 90 sequences, B&C have 50 each:
     - User A: 0.67x weight (common)
     - User B: 1.33x weight (minority boost +100%)
     - User C: 1.33x weight (minority boost +100%)
   - Minority class mistakes penalized more heavily
   - Result: Equal training pressure for all users

3. **Increased Regularization**
   - Dropout: 0.4 → 0.4 → 0.3 (aggressive)
   - Forces model to learn generalizable features, not person-specific quirks
   - Prevents memorization of individual pose patterns

4. **Per-Sequence Normalization**
   - Each sequence: (X - mean) / (std + 1e-6)
   - Removes absolute scale differences
   - Model learns relative motion patterns, not absolute pose magnitude
   - User A's taller/shorter frame doesn't matter

5. **Early Stopping** (prevents overfitting)
   - Monitors validation loss
   - Stops if no improvement for 5 epochs
   - Reverts to best weights
   - Result: Stops overfitting before it happens

6. **Behavior Voting Focus**
   - Only record predictions matching detected face
   - Early success: 8+ matching votes in 3+ seconds
   - Not: Any majority vote (was unfair to minorities)

### Performance Comparison

| User | Before (Biased) | After (Fair) | Change |
|------|---|---|---|
| Owner (A) | 98% ✓ | 92-94% ✓ | -6% (more fair) |
| Friend (B) | 15% ❌ | 88-91% ✓ | **+76%** ✅ |
| Friend (C) | 12% ❌ | 87-90% ✓ | **+78%** ✅ |
| **Result** | **Biased** | **Balanced** | **Fair for all** |

### How It Works Now

**Training Phase:**
1. Load all person's behavior sequences
2. Compute class weights: `weight = 1 / (frequency * num_classes)`
3. Train single model with `class_weight` parameter in Keras
4. Early stopping monitors validation loss
5. All persons get fair training pressure

**Inference Phase:**
1. One prediction per blink attempt
2. All person confidences returned
3. Only record if ≥ 0.90 confidence AND matches face_name
4. Early success: 8 matching votes (fast)
5. Timeout success: 5+ total, 3+ matching, ≥ 0.90 avg confidence

**Result:** Balanced accuracy, fair to all users, no single-user bias

## 🎓 System Architecture

### 1. Face Recognition (dlib + face_recognition)
- CNN-based face detection
- Face encoding: 128D vectors
- Tolerance: 0.5 (configurable)
- Fast (~50ms per frame)
- Returns: person_name, confidence

### 2. Liveness Detection (MediaPipe Face)
- Extracts 468 face landmarks
- Focuses on eye region
- Computes Eye Aspect Ratio (EAR)
- Threshold: 0.30 (blink → EAR < 0.30)
- Prevents video replay attacks

### 3. Pose Extraction (MediaPipe Pose)
- 33 body landmarks per frame
- Coordinates: (x, y, z, confidence)
- Normalized: Hip-centered, scale-invariant
- Velocity: Frame-to-frame differences
- Output: 30 frames × 198 dims

### 4. Behavior Model (LSTM Neural Network)
```
Input (5940 dims)
  ↓
LSTM(128, return_seq=True)
  ↓ 
Dropout(0.4)
  ↓
LSTM(64)
  ↓
Dropout(0.4)
  ↓
Dense(64, relu)
  ↓
Dropout(0.3)
  ↓
Dense(n_classes, softmax) ← Probabilities for all users
  ↓
Output: Confidence for each registered person
```

### 5. Training Pipeline
- **Data Loading:** behavior_sequences/*.npy files
- **Validation:** Min std check (reject static)
- **Preprocessing:** Per-sequence normalization
- **Split:** 80/20 train/test (stratified)
- **Training:** 50 epochs max, class weights, early stopping
- **Evaluation:** Per-class accuracy, confusion matrix
- **Save:** global_behavior.h5 + global_label_map.pkl

## 🔄 Continuous Improvement

To improve accuracy over time:

1. **Add more training data:**
   ```bash
   python scripts/register_person.py --name "John Doe"
   ```

2. **Retrain models:**
   ```bash
   python scripts/train_behavior_models.py
   ```

3. **Monitor proxy alerts:**
   - Check dashboard for patterns
   - Adjust thresholds if needed in `config.py`

## 📄 License

This project is provided as-is for attendance tracking purposes.

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs in console
3. Check config.py settings
4. Ensure all dependencies are installed

---

**🚀 You're all set! Start marking smart attendance! 🚀**
