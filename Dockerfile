# exercise 1 - Python virtual environment setup and test
# @authors: nobodycaresdude with help from Claude Opus 5.5
# uses a distroless Python image
FROM python:3.12-slim

# set up the working directory and copy the application code
WORKDIR /app
COPY app.py .

# specify the command to run the application
CMD ["python3", "app.py", "parameter1"]