Master Project


cd c:\Users\david\Desktop\project

docker build -t pcb-defect-classifier -f notebook/Dockerfile .

docker run --rm -v c:\Users\david\Desktop\project\dataset:/app/dataset pcb-defect-classifier /app/dataset/Labeled/WIN_20220330_16_02_56_Pro.jpg

