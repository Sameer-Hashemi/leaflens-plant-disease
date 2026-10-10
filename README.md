# LeafLens — Plant Disease Detection

A Streamlit web app for your trained EfficientNetB0 plant-disease classifier. The app uses `best_model.keras` and `class_names.json` from the supplied training output.

## Project files

- `app.py` — web application and prediction interface
- `best_model.keras` — trained model (~29.5 MB)
- `class_names.json` — class labels and saved experiment metrics
- `requirements.txt` — pinned Python dependencies for deployment

## Deploy publicly with Streamlit Community Cloud

1. Test locally first (instructions below).
2. Create a GitHub repository named `leaflens-plant-disease`.
3. Upload the four project files above to the **root** of the repository. Do not upload the original ZIP or experiment figures unless you want them in the repository.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app** / **Deploy an app**, select the repository and branch, and set the main file path to `app.py`.
6. Deploy. The first build can take several minutes because TensorFlow is a large dependency. If the build fails, open the app's build logs and check the dependency installation error.
7. Open the generated `*.streamlit.app` URL and test several leaf images.

## Test on Windows

Use Python 3.11 (64-bit) as a starting point. In PowerShell, from this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Open http://localhost:8501.

## Important notes

- Keep `app.py`, `best_model.keras`, and `class_names.json` in the same directory.
- This model expects RGB images resized to 224 × 224, with pixel values in the 0–255 range; preprocessing is embedded in the saved model according to the metadata.
- The saved test metrics are from an in-domain PlantVillage test split. They should not be represented as guaranteed real-world field accuracy.
- The Grad-CAM explanation is best-effort and may be unavailable depending on the model's internal layer structure. The classification prediction does not depend on Grad-CAM.
- A public repository exposes the model file for download. Use a private repository only if your hosting plan and deployment workflow support it.
- Free hosting resources and dependency compatibility can change. If deployment fails, inspect the build logs before changing TensorFlow versions.
