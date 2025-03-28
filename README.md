# Sleep Health and Lifestyle Dashboard

![Dashboard Screenshot](https://github.com/user-attachments/assets/a2442f24-89f8-4193-b410-98d25ae12330)

A data visualization dashboard exploring the relationship between sleep quality, lifestyle factors, and health metrics. This interactive web application provides insights into how various factors like physical activity, stress levels, and occupational characteristics affect sleep quality and overall health.

## Live Demo

The dashboard is deployed on Google Cloud Run and accessible at:
[Sleep Analysis Interactive Dashboard](https://viz-dashboard-408143638721.us-central1.run.app/)

## Dataset

This project uses the [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data) from Kaggle, which contains information about:

- **Demographics**: Gender, Age, Occupation
- **Sleep metrics**: Duration, Quality, Sleep Disorders
- **Health indicators**: BMI, Blood Pressure, Heart Rate
- **Lifestyle factors**: Physical Activity, Stress Level, Daily Steps

The dataset includes 374 individuals with diverse demographic and health profiles, allowing for comprehensive analysis of sleep health correlations.

## Features

- **Interactive Visualizations**: Explore distributions and correlations through dynamic charts
- **Multiple Analysis Categories**:
  - Distributions of key variables
  - Feature correlations (relationships between input variables)
  - Target correlations (relationships between sleep metrics)
  - Feature-target relationships
- **Theme Toggle**: Switch between light and dark themes for comfortable viewing
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Data Insights**: Explanatory text highlighting key findings

## Key Findings

- **Sleep Duration & Quality**: Strong positive correlation between sleep duration and perceived quality
- **Stress Impact**: Higher stress levels correlate with decreased sleep quality and duration
- **Occupation Patterns**: 
  - Healthcare professionals (doctors, nurses) show varying sleep patterns and stress levels
  - Sales professionals report higher stress levels and poorer sleep quality
- **Physical Activity**: Regular physical activity correlates with better sleep quality
- **Age Factors**: Stress levels tend to decrease after age 50
- **BMI Correlation**: People with higher BMI categories show increased risk of sleep disorders
- **Cardiovascular Health**: Regular physical activity (6,000-10,000 steps daily) correlates with better heart rate metrics

## Technology Stack

- **Python**: Core programming language
- **Dash & Plotly**: For creating interactive visualizations and web interface
- **Pandas & NumPy**: For data manipulation and analysis
- **SciPy**: For statistical operations like kernel density estimation
- **Google Cloud Run**: For deployment and hosting
- **Docker**: For containerization

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Local Setup
1. Clone this repository:
```bash
git clone https://github.com/Emine-Bassoum/Sleepy-Dashboard-Project.git
cd Sleepy-Dashboard-Project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to http://127.0.0.1:8080 or the address shown in your terminal

## Project Structure

```
Sleepy-Dashboard-Project/
│
├── app.py                      # Main dashboard application
├── Sleep_health_and_lifestyle_dataset.csv  # Dataset file
├── Visualisation_de_donnees_graphiques.ipynb  # Data exploration notebook
├── assets/
│   └── styles.css              # Dashboard styling
├── docs/
│   ├── Diaporama visualisation donnees.pdf  # Presentation slides
│   └── Rapport_visualisation_de_donnes.pdf  # Project report (French)
├── .gcloudignore               # Files to ignore in Google Cloud deployment
├── cloudbuild.yaml             # Google Cloud build configuration
├── Dockerfile                  # Docker container configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

Navigate through the dashboard tabs to explore different aspects of sleep health:

1. **Distributions**: View the distribution of individual variables
2. **Features Correlations**: Analyze relationships between lifestyle factors
3. **Targets Correlations**: Examine relationships between sleep metrics
4. **Features/Targets Relations**: Investigate how lifestyle factors affect sleep outcomes

## Deployment

The project is deployed using Google Cloud Run with continuous deployment set up through Cloud Build. The configuration is defined in the `cloudbuild.yaml` file.

To deploy manually:

1. Build the Docker container:
```bash
docker build -t sleepy-dashboard .
```

2. Run the container locally for testing:
```bash
docker run -p 8080:8080 sleepy-dashboard
```

3. Deploy to Google Cloud Run (requires Google Cloud SDK):
```bash
gcloud run deploy viz-dashboard --image gcr.io/YOUR_PROJECT_ID/viz-dashboard:latest --platform managed --allow-unauthenticated
```

## Authors

- BASSOUM Mohamed Emine
- GUEMIMI Marouane
- WIATT Chloé
- AUTEF Lucas
- JEANNE Arthur

## Repository

Source code is available at:
[GitHub Repository](https://github.com/Emine-Bassoum/Sleepy-Dashboard-Project)

## Acknowledgments

- Dataset provided by Kaggle
- Built with Dash and Plotly libraries
- Analysis conducted as part of data visualization project
