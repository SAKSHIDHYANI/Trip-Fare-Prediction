FROM python:3.8.8

WORKDIR /app/
EXPOSE 5000
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "app.py"]