# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install dependencies: git and git-lfs (needed to pull LFS files)
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Set environment variable to prevent buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the entire repository into the container
# Make sure that your .dockerignore does NOT exclude the .git folder
COPY . .

# Run Git LFS pull to fetch large files
RUN git lfs pull

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
