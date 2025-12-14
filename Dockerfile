
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /work

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4

# Update the package lists
RUN apt-get update
# Install the required libGL dependency (usually in libgl1-mesa-glx or similar)
RUN apt-get install -y libgl1-mesa-glx

COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create a directory for data (to be mounted)
RUN mkdir -p /work/data
RUN chmod +x /work/run.sh || true

# Set the entrypoint to run the training script by default
# You can override this with `docker run ... python src/04-inference.py` etc.
CMD ["bash", "/work/run.sh"]