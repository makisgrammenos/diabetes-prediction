FROM python:3.10

RUN adduser --disabled-password --gecos '' appuser

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "--server.enableCORS", "false", "app.py"]
