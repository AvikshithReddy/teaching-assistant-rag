FROM python:3.11-slim

WORKDIR /app

# System deps (optional for performance / builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Download required NLTK data inside the image
RUN python -m nltk.downloader stopwords punkt wordnet

# Copy the rest of the app
COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "main_app.py"]