# 📊 Football Match Outcome Prediction  
Data Science and Advanced Programming 

1. Project Overview

This project predicts English Premier League football match outcomes (Home Win, Draw, Away Win) using supervised machine learning techniques.  
It combines automated data collection, feature engineering, multiple predictive models, and financial interpretation, with a strong emphasis on reproducibility, modularity, and clean software design.

The entire pipeline is containerized using Docker, ensuring consistent execution across different systems and full reproducibility of results.

2. Project Structure

Project directory structure:

.
- data_loader.py          : Download and preprocess raw match data
- models.py               : Train and tune machine learning models
- evaluation.py           : Evaluate performance and generate plots
- notebooks/
  - exploration.ipynb     : Exploratory data analysis and experiments
- figures/                : All figures used in the report (PNG)
- Dockerfile              : Docker environment definition
- requirements.txt        : Python dependencies
- README.md               : Project documentation
- main.py                 : Optional entry point to run the full pipeline

3. Requirements

The project can be run either using Docker (recommended) or locally without Docker.

Docker option:
- Docker installed (Docker Desktop on Windows/macOS or Docker Engine on Linux)

Local option:
- Python version 3.9 or higher
- pip package manager

Local dependency installation:
pip install -r requirements.txt

4. Running the Project with Docker

Build the Docker image from the project root:
docker build -t football-ml .

Run the full pipeline:
docker run --rm football-ml

This command automatically:
- Downloads match data
- Cleans and preprocesses the dataset
- Engineers enriched features
- Trains all machine learning models
- Evaluates performance
- Generates figures (confusion matrices, feature importance, model comparison)

All results are deterministic thanks to fixed random seeds.

5. Running the Project without Docker

Each step can be executed manually:

Data collection and preprocessing:
python data_loader.py

Model training:
python models.py

Model evaluation:
python evaluation.py

6. Notebooks

The notebooks contains notebooks used for:
- Exploratory data analysis
- Feature inspection
- Intermediate experimentation and validation

7. Reproducibility

Reproducibility is ensured through:
- Fixed random seeds across all models
- Strict temporal train–validation–test split to avoid data leakage
- Docker containerization to guarantee identical environments across machines

This ensures that all results reported in the paper can be reproduced exactly.

8. Figures and Report

All figures used in the final report (confusion matrices, model performance comparison, feature importance, pipeline diagram) are stored in the figures directory.

The research paper is written in LaTeX using an academic conference style (SIAM / IEEE).

9. Use of External Tools

Generative AI tools such as ChatGPT were used selectively to assist with:
- Implementing complex parts of the code
- Improving clarity and structure of the written report

All modeling decisions, feature engineering, experimental design, and interpretation of results were conceived and implemented by the author.

10. Author

Alexandre Levenishti  
MSc in Finance – HEC Lausanne  
Course: Data Science and Advanced Programming

