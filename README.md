# 🌿 LeafLens --- AI-Powered Plant Disease Detection

LeafLens is a computer vision web application that screens plant-leaf
images for potential diseases using a trained **EfficientNetB0**
image-classification model. Upload a leaf photo to receive the model's
top prediction, confidence score, top-three predictions, and an optional
Grad-CAM visual explanation.

> **Research prototype:** LeafLens is an educational screening aid, not
> a definitive agronomic diagnosis. Predictions can be unreliable on
> unfamiliar crops, field photographs, or images that differ from the
> model's training data.

## ✨ Features

-   **Leaf image upload** --- supports JPG, JPEG, and PNG images.
-   **AI prediction** --- classifies the uploaded image using the
    trained model.
-   **Confidence score** --- displays the model score for the top
    prediction.
-   **Top-three results** --- compares the three highest-scoring
    classes.
-   **Grad-CAM explanation** --- optionally highlights image regions
    that may have influenced the prediction, when supported by the
    model.
-   **CSV export** --- download the top-three prediction summary.
-   **Model information** --- displays model architecture, input size,
    and supported classes.
-   **Hosted model weights** --- downloads `best_model.keras` from
    Hugging Face when the file is not available locally.

## 🧰 Tech Stack

-   Python
-   Streamlit
-   TensorFlow / Keras
-   EfficientNetB0
-   NumPy and pandas
-   Pillow
-   OpenCV (headless)
-   Requests
-   GitHub for source code
-   Hugging Face for model storage
-   Streamlit Community Cloud for deployment

## 🧠 Model Details

  -----------------------------------------------------------------------
  Item                                Details
  ----------------------------------- -----------------------------------
  Architecture                        EfficientNetB0

  Input size                          224 × 224 pixels

  Output                              Classification across the classes
                                      listed in `class_names.json`

  Model file                          `best_model.keras`

  Model hosting                       Hugging Face Hub
  -----------------------------------------------------------------------

The class labels and associated metadata are loaded from
`class_names.json`. The application expects the model's output count to
match the number of class names.

## 🚀 Live Demo

**Live application:**
https://leaflens-plant-disease-4xumwy3sjeebn6u8aawwn.streamlit.app/

## 📁 Repository Structure

``` text
leaflens-plant-disease/
├── app.py
├── class_names.json
├── requirements.txt
└── README.md
```

The trained model file is hosted separately on Hugging Face to avoid
storing the large model in the GitHub repository.

## ⚙️ How the App Loads the Model

1.  Streamlit starts `app.py`.
2.  The app checks whether `best_model.keras` is available locally.
3.  If it is missing, the app downloads the model from the configured
    Hugging Face repository.
4.  The app loads the model and reads class metadata from
    `class_names.json`.
5.  When an image is uploaded, it is converted to RGB and resized to 224
    × 224 pixels.
6.  The model predicts class scores, and the app displays the top
    prediction and top three classes.
7.  If enabled and supported, Grad-CAM generates a visual explanation.

## 🖥️ Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_GITHUB_USERNAME/leaflens-plant-disease.git
cd leaflens-plant-disease
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment (recommended)

**Windows:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Start Streamlit

``` bash
streamlit run app.py
```

The terminal will provide a local URL to open in your browser. The
trained model must be accessible at the configured Hugging Face URL or
present locally, and `class_names.json` must be available beside
`app.py`.

## ☁️ Deployment

This project is deployed with **Streamlit Community Cloud**.

1.  Push `app.py`, `class_names.json`, `requirements.txt`, and
    `README.md` to the GitHub repository.
2.  Keep the model weights in the configured Hugging Face repository.
3.  In Streamlit Community Cloud, create or select the app using the
    repository, `main` branch, and `app.py` as the entry point.
4.  Streamlit installs the dependencies from `requirements.txt` and
    starts the application.
5.  After committing code or dependency changes, allow the app to
    rebuild and check its logs if deployment fails.

## 📸 Tips for Better Predictions

-   Use a sharp, well-lit image.
-   Keep one leaf clearly visible and relatively large in the frame.
-   Avoid motion blur, heavy shadows, and extreme cropping.
-   Treat low-confidence results cautiously.
-   Confirm important crop-health decisions with a qualified
    agricultural specialist.

## ⚠️ Limitations

-   The model can only predict classes represented in its training
    labels.
-   A high confidence score does not guarantee that a prediction is
    correct.
-   Performance may decrease on field images, unfamiliar crops, unusual
    lighting, or backgrounds unlike the training data.
-   Grad-CAM is an interpretation aid, not proof that the model is
    correct.
-   LeafLens does not replace expert agricultural assessment.

## 👤 Author

**Ahmed Sameer Hashemi**\
BS Artificial Intelligence student

## 📄 Disclaimer

LeafLens is intended for educational and research demonstration
purposes. It should not be used as the sole basis for pesticide
application, crop treatment, or other consequential agricultural
decisions.
