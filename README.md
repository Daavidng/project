# PCB Defect Classification with Interactive RL

Simple and efficient PCB defect detection using Self-Supervised Few-Shot Reinforcement Learning.

## Features

- **Simple Output**: Clean classification results without verbose messages
- **Interactive RL**: Real-time adaptive learning with minimal human feedback
- **Edge-Ready**: Lightweight and optimized for deployment
- **New Class Discovery**: Automatically learns new defect types
- **No Special Characters**: Clean, compatible output for all systems

## Quick Start

### Setup
```bash
cd c:\Users\david\Desktop\project
pip install -r requirements.txt
```

### Usage

**Basic Classification:**
```bash
python run.py --image_path "dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg"
```

**Interactive RL Learning:**
```bash
python run.py --mode interactive --image_path "dataset\Labeled\WIN_20220329_14_30_32_Pro.jpg"
```

## Sample Output

**Basic Classification:**
```
Classes: ['exc_solder', 'good', 'poor_solder', 'spike']
good: 33.5%
exc_solder: 24.2%
poor_solder: 21.4%
spike: 20.9%
```

**Interactive RL Mode:**
```
Loaded 4 classes: ['exc_solder', 'good', 'poor_solder', 'spike']
Prediction: good (33.5%)
Model uncertain - feedback needed
Correct? (y/n/label): y
Reinforced 'good'
Save? (y/n): y
Saved to model/learned.pkl
```

## Available Classes
- **good**: Normal PCB (no defects)
- **exc_solder**: Excessive solder
- **poor_solder**: Insufficient solder
- **spike**: Solder spike defect

## How Interactive RL Works
1. **Analyzes uncertainty** using distance from known prototypes
2. **Requests feedback** only when model is uncertain
3. **Updates prototypes** based on human corrections
4. **Learns new classes** automatically during operation
5. **Saves improvements** for future use

## Technical Details

- **Model**: ResNet50-based SSL encoder with prototypical networks
- **Learning**: Few-shot learning with uncertainty-based RL feedback
- **Input**: 128x128 RGB images
- **Output**: Class probabilities and uncertainty scores
- **Deployment**: Edge-ready with minimal computational requirements

## Files Structure
- `run.py` - Main script with classification and interactive RL
- `model/ssl_encoder.weights.h5` - Pre-trained SSL encoder weights  
- `model/fsl_model_artifacts.pkl` - FSL prototypes and class names
- `notebooks/` - Training notebooks for SSL, FSL, and RL components