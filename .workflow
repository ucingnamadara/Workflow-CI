name: CI/CD MLflow

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  CSV_URL: "MLproject/dataset_preprocessing.csv"
  TARGET_VAR: "Genre"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE
      - uses: actions/checkout@v3
      
      - name: Set up Python 3.12.7
        uses: actions/setup-python@v4
        with:
          python-version: "3.12.7"

      - name: Check Env
        run: |
          echo $CSV_URL

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install mlflow

      - name: Run mlflow project
        run: |
          mlflow run MLproject --env-manager=local

      - name: Get latest MLflow run_id
        run: |
          RUN_ID=$(ls -td mlruns/0/*/ | head -n 1 | cut -d'/' -f3)
          echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV
          echo "Latest run_id: $RUN_ID"