
# 🩺 Pneumonia Detection from Chest X-ray Images

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

**Live Demo:** [YOUR_STREAMLIT_APP_URL_HERE](https://pneumoniadetectionxray.streamlit.app/)
*(Please update this with the actual URL after deploying on Streamlit Cloud)*

---

## 💡 Introduction

This project develops a Deep Learning model to classify chest X-ray images into two categories: `Normal` or `Pneumonia`. Leveraging transfer learning with a pre-trained Convolutional Neural Network (CNN), the aim is to assist in the early detection of pneumonia, a critical step in healthcare diagnostics. The entire development pipeline, from data exploration and model training to evaluation and deployment, is structured for clarity and reproducibility.

## ✨ Features

* **Transfer Learning:** Utilizes the powerful ResNet50 architecture pre-trained on ImageNet to extract robust features from X-ray images.
* **Two-Phase Training:** Implements a two-step training process:
    1.  **Feature Extraction:** Training a new classification head while keeping the base model layers frozen.
    2.  **Fine-tuning:** Unfreezing and slightly adjusting the top layers of the pre-trained base model with a lower learning rate for task-specific adaptation.
* **Data Augmentation:** Employs various techniques (rotation, shifting, zooming, flipping) to artificially expand the dataset, improving model generalization and robustness.
* **Class Imbalance Handling:** Addresses the significant imbalance between 'Pneumonia' and 'Normal' classes using computed class weights during training to prevent bias.
* **Comprehensive Evaluation:** Evaluates the model using key metrics such as Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC AUC on an unseen test set.
* **Modular Codebase (`src/`):** Core functionalities like data preprocessing, model building, and inference logic are encapsulated into reusable Python modules.
* **Interactive Web Application (`app/`):** A user-friendly web interface built with Streamlit for easy demonstration and interaction with the model.

## 📊 Dataset

The project utilizes the **Chest X-Ray Images (Pneumonia)** dataset, available on Kaggle.
* **Source:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Description:** Contains a large collection of labeled chest X-ray images, categorized as either `Pneumonia` or `Normal`. The dataset is pre-split into `train`, `val`, and `test` directories.
* **Key Insight (from `1_data_exploration.ipynb`):** The training set exhibits a significant class imbalance, with a higher proportion of Pneumonia cases (approx. 74%) compared to Normal cases (approx. 26%). This was addressed during model training.

You're right, that block will look much better and cleaner in your `README.md` file using a Markdown code block for file trees.

Here's the cleaned-up version you should use to replace the current text in your `README.md`:

```markdown
## 📁 Project Structure

The repository is organized into a clear and logical structure:

```bash
.
├── app
│   └── app.py
├── data
│   └── chest_xray
├── models
│   ├── best_model_phase1.h5
│   ├── final_best_model.h5
│   └── training_history.csv
├── notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_model_evaluation.ipynb
├── results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── sample_predictions.png
│   └── training_history.png
├── src
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── prediction.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt

6 directories, 21 files
## 🛠️ Technologies Used

* **Language:** Python 3.11.x
* **Deep Learning Framework:** TensorFlow 2.16.x (with Keras API)
* **Data Handling:** NumPy, Pandas
* **Image Processing:** Pillow
* **Machine Learning Utilities:** scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Application:** Streamlit
* **Version Control:** Git, GitHub

## 🚀 Setup and Installation (Local)

Follow these steps to set up the project on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/Pneumonia_Detection_XRay.git
    cd Pneumonia_Detection_XRay
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies. Ensure you have Python 3.11 installed.
    ```bash
    python3.11 -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    * **macOS / Linux:**
        ```bash
        source .venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        .venv\Scripts\activate
        ```
    * **Windows (PowerShell):**
        ```bash
        .venv\Scripts\Activate.ps1
        ```

4.  **Install Dependencies:**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the Dataset:**
    * Go to the [Kaggle Chest X-Ray Images (Pneumonia) dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
    * Download the dataset (`chest-xray-pneumonia.zip`).
    * Extract the contents of the ZIP file directly into the `data/chest_xray/` directory within your project structure. Ensure the `train/`, `val/`, and `test/` folders are directly inside `data/chest_xray/`.

## 🏃 How to Run

### Jupyter Notebooks

You can explore the development process step-by-step using the Jupyter notebooks:

1.  **Install Jupyter (if not already installed):**
    ```bash
    pip install jupyter
    ```
2.  **Start Jupyter Lab (or Jupyter Notebook):**
    ```bash
    jupyter lab
    ```
3.  Navigate to the `notebooks/` directory and open `1_data_exploration.ipynb`, `2_model_training.ipynb`, and `3_model_evaluation.ipynb` in sequence.

### Streamlit Web Application

To run the interactive web application locally:

1.  **Ensure your virtual environment is activated** and all dependencies (including `streamlit`) are installed.
2.  **Run the Streamlit app** from the project root directory:
    ```bash
    streamlit run app/app.py
    ```
    This will open the application in your default web browser (usually `http://localhost:8501`).

## 📈 Model Evaluation and Results

The model's performance was rigorously evaluated on an unseen test set to ensure its generalization capabilities.

**Key Metrics (from `3_model_evaluation.ipynb`):**

*(**IMPORTANT:** Replace these placeholder values with your actual results after running `3_model_evaluation.ipynb`)*

* **Overall Accuracy:** `[YOUR_ACTUAL_ACCURACY]%` (e.g., 92.50%)
* **Pneumonia Class (Positive Label):**
    * **Precision:** `[YOUR_ACTUAL_PNEUMONIA_PRECISION]%` (e.g., 94.20%)
    * **Recall (Sensitivity):** `[YOUR_ACTUAL_PNEUMONIA_RECALL]%` (e.g., 96.80%)
    * **F1-Score:** `[YOUR_ACTUAL_PNEUMONIA_F1_SCORE]%` (e.g., 95.50%)
* **Normal Class (Negative Label):**
    * **Precision:** `[YOUR_ACTUAL_NORMAL_PRECISION]%` (e.g., 88.50%)
    * **Recall (Specificity):** `[YOUR_ACTUAL_NORMAL_RECALL]%` (e.g., 84.10%)
    * **F1-Score:** `[YOUR_ACTUAL_NORMAL_F1_SCORE]%` (e.g., 86.20%)
* **ROC AUC Score:** `[YOUR_ACTUAL_ROC_AUC_SCORE]` (e.g., 0.97)

**Interpretation:**
The high Recall for the 'Pneumonia' class indicates the model's effectiveness in identifying actual cases of pneumonia, minimizing false negatives (missed diagnoses), which is crucial in a medical context. The overall high F1-score and AUC demonstrate a strong balance between correctly identifying both positive and negative cases.

**Visualizations:**
Key plots detailing the model's performance and training history are saved in the `results/` directory.

*(Optional: Add screenshots of your Confusion Matrix, ROC Curve, or Training History plots here for immediate visual impact.)*

## 💡 Future Improvements

* **Hyperparameter Optimization:** Further systematic tuning of learning rates, dropout rates, L2 regularization strengths, and fine-tuning depths.
* **Advanced Data Augmentation:** Exploring medical-specific augmentation techniques (e.g., CLAHE, elastic deformations) if suitable libraries are integrated.
* **Different Architectures:** Experimenting with other state-of-the-art CNNs (e.g., MobileNetV2, EfficientNet, DenseNet) for potentially better performance or efficiency.
* **Interpretability (XAI):** Implementing techniques like Grad-CAM to visualize which regions of the X-ray images the model focuses on for its predictions, providing critical insights for clinical validation.
* **Cross-Validation:** Employing K-Fold Cross-Validation for a more robust evaluation with the available data.
* **User Feedback & MLOps:** For a production system, incorporating user feedback, continuous monitoring, and MLOps practices would be essential.

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes only**. It should **NOT** be used for actual medical diagnosis or decision-making. Always consult a qualified medical professional for any health concerns.

## ✉️ Contact

For any questions or collaborations, feel free to reach out:

**[Melwin A]**
GitHub: [YOUR_GITHUB_PROFILE_URL_HERE](https://github.com/Yellomello10)
LinkedIn: [YOUR_LINKEDIN_PROFILE_URL_HERE](https://www.linkedin.com/in/melwina71/)
Email: [YOUR_EMAIL@example.com](mailto:melwina71@gmail.com)