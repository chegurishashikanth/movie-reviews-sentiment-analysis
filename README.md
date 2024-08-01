# Movie Reviews Sentiment Analysis

This project implements a Sentiment Analysis model using machine learning techniques to classify movie reviews as positive or negative. The goal is to understand the sentiments expressed in movie reviews and provide a predictive model that can automatically determine the sentiment of a given review.

## Features

- **Data Collection**: Utilizes a dataset of movie reviews with pre-labeled sentiments.
- **Data Preprocessing**: Includes steps like tokenization, stop word removal, and word embedding.
- **Model Training**: Implements various machine learning algorithms including Logistic Regression, Naive Bayes, and Support Vector Machines.
- **Model Evaluation**: Evaluates models using metrics such as accuracy, precision, recall, and F1-score.
- **Prediction**: Provides a simple interface to input a movie review and get the predicted sentiment.

## Technologies Used

- **Python**: Core programming language.
- **Pandas & Numpy**: For data manipulation and analysis.
- **Scikit-learn**: For implementing machine learning models.
- **NLTK**: For natural language processing tasks.
- **Jupyter Notebook**: For interactive development and experimentation.

## Project Structure

- `data/`: Contains the dataset of movie reviews.
- `notebooks/`: Jupyter notebooks with exploratory data analysis, model training, and evaluation.
- `scripts/`: Python scripts for data preprocessing, model training, and prediction.
- `models/`: Serialized models for easy loading and inference.
- `README.md`: Project overview and setup instructions.

## Setup Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/movie-reviews-sentiment-analysis.git
    cd movie-reviews-sentiment-analysis
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
    Place the dataset in the `data/` directory.

4. **Run the Jupyter notebooks**:
    Start Jupyter Notebook and open the notebooks in the `notebooks/` directory to explore data and train models.

## Usage

1. **Preprocess data**:
    ```sh
    python scripts/preprocess_data.py
    ```

2. **Train models**:
    ```sh
    python scripts/train_model.py
    ```

3. **Predict sentiment**:
    ```sh
    python scripts/predict_sentiment.py "Your movie review here"
    ```

## Results

- Detailed analysis and comparison of different models' performance.
- Visualization of model metrics and confusion matrices.
- Examples of correctly and incorrectly classified reviews.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
