# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the inference script and the model directory into the container
COPY notebook/inference.py .
COPY model ./model

# Copy the requirements file into the container at /app
COPY notebook/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define the entrypoint for the container.
# This command will be executed when the container starts.
ENTRYPOINT ["python", "inference.py"]
