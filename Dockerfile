# Use an official Python runtime as a parent image
FROM python:3.11 

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python","app.py" ]
