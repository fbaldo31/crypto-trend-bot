# Dockerfile to run the server
FROM python:latest

COPY ./Pipfile /Pipfile

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r /Pipfile

# Copy source files
COPY ./api /api
COPY ./server /server
COPY ./manage.py /manage.py
COPY ./start-server.sh /start-server.sh

# Start the server
CMD [ "/bin/bash", "./start-server.sh" ]
