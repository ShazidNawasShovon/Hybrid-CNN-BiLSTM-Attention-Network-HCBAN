# Use official TensorFlow GPU image
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and dataset
# Note: Ensure dataset folder is present before building
COPY . .

# Set environment variables for GPU optimization
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Default command to run the full research pipeline
CMD ["python", "src/research/research_pipeline.py"]
