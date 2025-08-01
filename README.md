# **🖊️ Digit Recognizer App (MNIST Handwritten Digit Classification)**

An interactive **Streamlit web app** that uses a trained **Convolutional Neural Network (CNN)** to recognize digits you draw on a canvas in real-time. Built and deployed locally using **Streamlit**.

---

## **Project Motivation**

This project was built as part of my journey through **Neural Networks** after completing several machine learning foundational projects (Linear and Logistic Regression). The goal was to:

* Learn and implement CNNs from scratch on MNIST

* Train, evaluate, and save a model

* Build an interactive web interface for users

* Deploy a fully functional ML app

---

## **Tech Stack**

* **Frontend/UI**: Streamlit

* **Model**: TensorFlow Keras CNN

* **Image Processing**: PIL (Pillow)

* **Data Handling**: NumPy, Pandas

* **Visualization**: Matplotlib (used in training phase)

---

## **Features**

* Digit input via mouse (drawable canvas)

* Real-time image preprocessing:

  * Grayscale conversion

  * Resize to 28x28

  * Normalization

* Live prediction with CNN

* Alerts for empty canvas

* Shows processed input (what the model actually sees)

---

## **How to Run Locally**

1. Clone the repo:  
   `git clone https://github.com/abdullah-codez/handwritten-digit-recognition-app`

2. Navigate into the folder:  
    `cd handwritten-digit-recognition-app`

3. Install dependencies:  
    `pip install -r requirements.txt`

4. Run the Streamlit app:  
    `streamlit run app.py`

---

## **Model Training (Behind-the-Scenes)**

* Dataset: MNIST

* Architecture: Simple CNN  
   (Conv2D → ReLU → MaxPooling → Flatten → Dense)

* Trained with:

  * Categorical Crossentropy Loss

  * Adam Optimizer

  * Accuracy \> 98% on test set

Model training and evaluation details are available in the Jupyter notebook.

---

## **About Me**

I'm **Muhammad Abdullah Iftikhar**, a CS undergrad at PIEAS, passionate about Machine Learning, building real-world ML/AI projects, and sharing my journey. If you like my work, consider giving this repo a star ⭐, Thank you.

Connect with me:

GitHub: [github.com](https://github.com/abdullah-codez)

LinkedIn: [linkedin.com](https://www.linkedin.com/in/muhammad-abdullah-iftikhar-543478361/)

