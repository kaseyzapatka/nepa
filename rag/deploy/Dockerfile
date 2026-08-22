FROM python:3.11-slim

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/app.py /app/app.py
COPY src /app/src

ENV PYTHONPATH=/app/src
EXPOSE 7860

CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
