# Sleep Health Dashboard

A visualization dashboard for sleep health data built with Plotly Dash.

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   python app.py
   ```
   
3. Open http://127.0.0.1:8080 in your browser

## Google Cloud Deployment

### Manual Deployment

1. Build the Docker image:
   ```
   docker build -t gcr.io/[PROJECT_ID]/viz-dashboard .
   ```

2. Push to Container Registry:
   ```
   docker push gcr.io/[PROJECT_ID]/viz-dashboard
   ```

3. Deploy to Cloud Run:
   ```
   gcloud run deploy viz-dashboard --image gcr.io/[PROJECT_ID]/viz-dashboard --platform managed --allow-unauthenticated
   ```

### Automatic Deployment with Cloud Build

1. Connect your GitLab repository to Google Cloud Build
2. Create a trigger that uses the cloudbuild.yaml configuration
3. Push changes to your repository to trigger automatic builds and deployments

## Data

The dashboard uses the Sleep Health and Lifestyle dataset, which is already included in the repository.

## Made with ❤️ by
- Mohamed Emine BASSOUM
- Marouane Guemimi
- Chloé Wiatt
- Lucas Autef
- Arthur Jeanne