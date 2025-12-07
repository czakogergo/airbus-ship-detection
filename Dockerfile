
FROM tensorflow/tensorflow:2.20.0-gpu-jupyter

WORKDIR /work

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create a directory for data (to be mounted)
RUN mkdir -p /work/data
RUN chmod +x /work/run.sh || true

# Set the entrypoint to run the training script by default
# You can override this with `docker run ... python src/04-inference.py` etc.
# CMD ["bash", "/work/run.sh"]