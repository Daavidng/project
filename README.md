# Self-Supervised Few-Shot Reinforcement Learning for PCB Defect Detection

## Features

- **Interactive RL Learning**: Real-time adaptive learning with human-in-the-loop feedback
- **Edge-Ready Deployment**: Scalable and adaptive anomaly detection for Industrial IoT
- **Clean, Simple Output**: Streamlined classification results showing all classes with probabilities
- **Suppressed TensorFlow Warnings**: Clean console output without verbose TensorFlow messages
- **Uncertainty-Based Feedback**: Only requests human input when model is uncertain
- **New Class Discovery**: Automatically learns new anomaly types during operation
- **Adaptive Thresholds**: Self-tuning parameters that improve with experience

## Quick Setup

### Option 1: Python Direct Usage (Recommended)

#### Step 1: Navigate to Project Directory
```bash
cd c:\Users\david\Desktop\project
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Run Inference

**Single Image Classification:**
```bash
python run.py --mode classify --image_path "dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg"
```

**Interactive RL Learning (Recommended for Industrial IoT):**
```bash
python run.py --mode interactive --image_path "dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg"
```

### Option 2: Docker Setup

#### Step 1: Build Docker Image (One-time setup)
```bash
docker build -t pcb-defect-classifier .
```

#### Step 2: Run Docker Container

**Single Image Classification:**
```bash
docker run --rm -v "C:\Users\david\Desktop\project\dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg:/app/image.jpg" -v "C:\Users\david\Desktop\project\model:/app/model" pcb-defect-classifier --mode classify --image_path /app/image.jpg
```

**RL-based Active Learning:**
```bash
docker run --rm -v "C:\Users\david\Desktop\project\dataset\Labeled:/app/images" -v "C:\Users\david\Desktop\project\model:/app/model" pcb-defect-classifier --mode active_learning --unlabeled_dir /app/images --steps 5 --samples_per_step 3
```

## Sample Output

**Classification Mode:**
```
Loading SSL+FSL model...
Model loaded successfully
Available classes: ['exc_solder', 'good', 'poor_solder', 'spike']

Classifying: dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg

=== Classification Report ===
good (33.51%)
exc_solder (24.23%)
poor_solder (21.37%)
spike (20.89%)
```

The classification report shows all defect classes ranked by probability, making it easy to see the model's confidence across all possible outcomes.

**Interactive RL Learning Mode:**
```
Loading Interactive RL Model for Edge Deployment...
Model loaded with classes: {0: 'exc_solder', 1: 'good', 2: 'poor_solder', 3: 'spike'}
Interactive RL with Uncertainty Sampling ready!
Initial thresholds - Uncertainty: 1000.0, Confidence: 0.4

Classifying: dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg

=== Classification Report ===
good (33.51%)
exc_solder (24.23%)
poor_solder (21.37%)
spike (20.89%)

Model Uncertainty Score: 11676096.00
Top Prediction Confidence: 33.5%
🤔 Model is uncertain - requesting human feedback...
Predicted: good
Is this correct? (y/n) or enter correct label: y
✅ Correct! Reinforcing this prediction...
✅ Reinforced 'good' class with this example

📊 Learning Progress:
Total classes: 4
Classes: ['exc_solder', 'good', 'poor_solder', 'spike']
Learning interactions: 1

Save learned model? (y/n): y
💾 Saved updated model to: model/learned_model.pkl
   Total classes learned: 4
   Learning interactions: 1
```

## How Interactive RL Works (Stage 2 - Edge-Ready Industrial IoT)

The Interactive RL system provides **real-time adaptive learning** perfect for edge-based industrial IoT systems:

### 🤖 **RL Agent Intelligence:**
1. **Uncertainty Analysis**: Calculates model confidence using distance from prototypes
2. **Smart Feedback Requests**: Only asks humans for help when truly uncertain
3. **Adaptive Thresholds**: Self-tunes decision boundaries based on feedback quality
4. **New Class Discovery**: Automatically learns new industrial anomaly types
5. **Edge Optimization**: Lightweight updates without full model retraining

### 📊 **Industrial IoT Benefits:**
- **Scalable**: Runs efficiently on edge devices with limited resources
- **Adaptive**: Continuously improves with minimal human intervention  
- **Real-time**: Provides instant classification with selective human feedback
- **Cost-effective**: Minimizes expert annotation time through intelligent sampling
- **Self-improving**: Adapts thresholds and learns new anomaly patterns automatically

### 🏭 **Perfect for Industrial Settings:**
- **Quality Control**: Learns new defect patterns as manufacturing evolves
- **Anomaly Detection**: Discovers unexpected failure modes in real-time
- **Expert Integration**: Leverages domain expert knowledge when model is uncertain
- **Continuous Learning**: Maintains performance as new products/processes are introduced

### 🔧 **Interactive RL Features:**
- **Uncertainty Sampling**: Uses same method as training notebook for consistency
- **Threshold Adaptation**: Becomes more selective as it learns from feedback
- **Prototype Updates**: Incremental learning without storing raw training data
- **Learning History**: Tracks interactions for performance analysis
- **Model Persistence**: Saves learned knowledge for deployment continuity

The Interactive RL mode represents the **practical deployment** of your research topic: "Self-Supervised Few-Shot Reinforcement Learning for Scalable and Adaptive Anomaly Detection in Edge-based Industrial IoT Systems".

## Available Defect Classes

The model can detect 4 types of PCB defects with probability scores for each:
- **good**: No defect (normal PCB)
- **exc_solder**: Excessive solder
- **poor_solder**: Insufficient solder  
- **spike**: Solder spike defect

## Recent Improvements

- **Interactive RL Integration**: Replaced batch active learning with real-time adaptive learning
- **Edge-Ready Deployment**: Optimized for industrial IoT edge devices
- **Uncertainty-Based Decisions**: Only requests human feedback when model is uncertain
- **New Class Discovery**: Automatically learns new defect types during operation
- **Adaptive Thresholds**: Self-tuning parameters that improve with experience
- **Enhanced Output**: Shows probability scores and uncertainty metrics
- **Code Optimization**: Streamlined for production deployment
- **Cleaner Console**: Suppressed TensorFlow verbose messages for better user experience