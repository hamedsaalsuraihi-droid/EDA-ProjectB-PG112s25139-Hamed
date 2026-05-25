# Mini Project B - Time-Series Forecasting Starter

Student: Hamed Alsuraihi  
Student ID: PG112s25139

This repository is a starter for UTAS Energy Data Analytics Mini Project B.

## Files

- `app.py` - one-file Streamlit app
- `requirements.txt` - required Python packages
- `data/dataset_sample.csv` - cleaned/sorted dataset slice, maximum 250,000 rows

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a public GitHub repository named `EDA-ProjectB-PG112s25139`.
2. Upload these files exactly:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Open Streamlit Community Cloud.
4. Choose New app.
5. Connect the GitHub repo.
6. Use branch `main`.
7. Set the main file path to `app.py`.
8. Deploy.

## OpenRouter API key

The app does not hardcode any API key. It reads the key from:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field in the app UI

## What to submit

Submit:

- Streamlit app URL
- GitHub repo URL
- exported `submission.json`
- exported `project_card.md`
- screenshots required by your instructor
