FROM your-base-container

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Copy the fine-tuned model files
COPY fine-tuned-gpt/ ./fine-tuned-gpt/


# Define the command to run the app
CMD ["python", "app.py"]
