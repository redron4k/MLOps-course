FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY hw11/requirements-runtime.lock /app/requirements-runtime.lock
RUN pip install --no-cache-dir -r requirements-runtime.lock
COPY hw8910/release.py /app/release.py
COPY hw8910/service.py /app/previous_service.py
COPY hw11/server.py hw11/client.py hw11/backend.py /app/
COPY hw11/artifacts /app/artifacts
USER 10001:10001
EXPOSE 8000
CMD ["python", "server.py"]
