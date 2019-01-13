# Dockerfile to run the server
FROM tensorflow/tensorflow:nightly-py3-jupyter

COPY ./Pipfile /Pipfile

# Install dependencies
RUN pip install --upgrade pip setuptools && \
    pip install -r /Pipfile

# Copy source files
COPY ./api /api
COPY ./server /server
COPY ./manage.py /manage.py
COPY ./start-server.sh /start-server.sh

EXPOSE 8080

# Start the server
CMD [ "/bin/bash", "/start-server.sh" ]
