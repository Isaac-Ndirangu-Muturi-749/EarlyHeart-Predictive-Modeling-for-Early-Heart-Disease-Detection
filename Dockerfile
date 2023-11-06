# Use the Python 3.8.12 image as the base image
FROM python:3.8.12-slim

# Install pipenv
RUN pip install pipenv

# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock to the container
COPY Pipfile Pipfile.lock ./

# Install the Python dependencies using pipenv
RUN pipenv install --system --deploy

# Copy the application files to the container
COPY predict.py random_forest_model.pkl ./

# Expose port 9696
EXPOSE 9696

# Set the entry point to run your application with Waitress
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
