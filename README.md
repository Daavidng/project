# SELF-SUPERVISED FEW-SHOT REINFORCEMENT LEARNING FOR SCALABLE AND ADAPTIVE ANOMALY DETECTION IN EDGE-BASED INDUSTRIAL IOT SYSTEMS

## Quick Setup

### Step 1: Navigate to Project Directory

```
cd c:\Users\david\Desktop\project
```

### Step 2: Build Docker Image - 1 time job

```
docker build -t pcb-defect-classifier -f notebook/Dockerfile .
```

### Step 3: Run Docker Container

```
docker run --rm `
  -v c:\Users\david\Desktop\project\dataset:/app/dataset `
  -v c:\Users\david\Desktop\project\notebook\inference.py:/app/inference.py `
  -v c:\Users\david\Desktop\project\model:/app/model `
  pcb-defect-classifier /app/dataset/Labeled/YOUR_IMAGE.jpg
```

```
docker run --rm `
  -v c:\Users\david\Desktop\project\dataset:/app/dataset `
  -v c:\Users\david\Desktop\project\notebook\inference.py:/app/inference.py `
  -v c:\Users\david\Desktop\project\model:/app/model `
  pcb-defect-classifier /app/dataset/Labeled/WIN_20220330_16_02_56_Pro.jpg
```

## Sample Output

```
Loading model...
Model loaded successfully.
Classifying image: /app/dataset/Labeled/WIN_20220330_16_02_56_Pro.jpg
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step

--- Inference Result ---
{'predicted_class': 'spike', 'confidence': 1.9683484424604103e-06, 'class_index': 3}
```