# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install all required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Expose port 8501 for Streamlit (if using)
EXPOSE 8501

# Default command: Run the integrated Python script
# If you want the terminal workflow, keep this:
CMD ["python", "main.py"]

# If you want to run the Streamlit web app instead, replace with:
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
