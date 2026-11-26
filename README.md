# Wine Quality Chemistry Explorer & Grader

An educational machine learning web application for exploring physicochemical features in red and white wine and estimating the quality score using a Random Forest regression model.

## Features

- Red and white wine selection
- Inputs for all 11 physicochemical features
- Random Forest quality-score prediction
- Simple project-defined grade labels
- Dataset exploration and interactive charts
- SQLite database for source records and prediction history
- Model evaluation using MAE and R²
- CSV export of prediction history

## Tech stack

Python, Streamlit, Pandas, scikit-learn, Plotly, SQLite, Joblib.

## Dataset

The app downloads the red and white CSV files from the UCI Machine Learning Repository on first run:

https://archive.ics.uci.edu/dataset/186/wine+quality

The source dataset contains physicochemical measurements and human-assigned sensory quality scores. It is intended for machine learning education, not medical, regulatory, or professional wine assessment.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open the VS Code terminal.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install packages:

   ```powershell
   pip install -r requirements.txt
   ```

6. Train the model:

   ```powershell
   python train_model.py
   ```

7. Start the website:

   ```powershell
   streamlit run app.py
   ```

8. Streamlit will show a local URL, usually `http://localhost:8501`.

The first run needs an internet connection to download the dataset. The SQLite database and trained model are generated locally.

## Quality grade mapping

The app converts the rounded predicted score into illustrative labels:

- 0–4: Low
- 5: Average
- 6: Good
- 7–10: Very good

These labels are a simple application design choice, not an official grading or certification standard.

## GitHub

Before pushing to GitHub, review `.gitignore`. The downloaded dataset files, SQLite database, and model artifacts are excluded to keep the repository lightweight. The app can recreate them when it runs with internet access.

```powershell
git init
git add .
git commit -m "Build wine quality chemistry explorer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL.

## Limitations

- The model learns patterns in the supplied historical dataset; its prediction is an estimate.
- The model uses the 11 chemical measurements. Wine color is recorded separately as metadata.
- The grade labels are illustrative.
- Do not interpret model output as a laboratory measurement or guarantee of quality.
