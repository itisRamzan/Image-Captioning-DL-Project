# 📢 Image Captioning with Flickr8k Dataset

This project implements an Image Captioning system using the Flickr8k dataset and deep learning techniques. The system generates descriptive captions for images using a combination of Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) with LSTM units for sequence generation.

## 🚀 Project Structure

The project is structured as follows:

- **Import Modules:** Python modules for data preprocessing, model building, and evaluation.
- **Extract Image Features:** Uses a pre-trained VGG16 model to extract image features and stores them as pickle files.
- **Load the Captions Data:** Loads and preprocesses captions associated with the images in the dataset.
- **Preprocess Text Data:** Cleans and preprocesses text data by tokenizing and padding sequences.
- **Train Test Split:** Splits the data into training and testing sets for model training.
- **Model Creation:** Defines and trains the deep learning model for image captioning.
- **Generate Captions for the Image:** Generates captions for new images and evaluates using BLEU scores.
- **Visualize the Results:** Visualizes actual vs. predicted captions for sample images.

## ⬇️ How to Use

1. **Setup Environment:**
   - Ensure Python environment is set up with necessary dependencies (`requirements.txt`).
   - Install required libraries: TensorFlow, tqdm, NLTK, PIL, matplotlib, etc.

2. **Dataset Preparation:**
   - Download the Flickr8k dataset and arrange images with associated captions.
   - Ensure `captions.txt` file is formatted correctly with image IDs and captions.

3. **Model Training:**
   - Run the script to extract image features and preprocess captions.
   - Train the model using the prepared data, adjusting hyperparameters as necessary.
   - Monitor training progress and save the best performing model (`best_model.h5`).

4. **Generate Captions:**
   - Use the trained model to generate captions for new images.
   - Visualize the results and evaluate caption quality using BLEU scores.

## 🎯 Evaluation

- **BLEU Score:** Evaluates the quality of generated captions against reference captions using the BLEU metric (BiLingual Evaluation Understudy).

## 🛠️ Dependencies

Ensure you have the following Python libraries installed:

```python
numpy
tensorflow
tqdm
pillow
matplotlib
nltk
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## 👉 Contact

For any inquiries or feedback, please reach out to:
- **Name**: Mohd Ramzan Shareef
- **Email**: mail.ramzanshareef@gmail.com
- **GitHub**: [ramzanshareef](https://github.com/ramzanshareef)
