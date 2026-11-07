# 📚 Amazon Bestselling Books: Data Science & Machine Learning Analysis

## 📌 Project Overview

This project explores the Amazon Top 50 Bestselling Books dataset from 2009 to 2019 using Python.

The goal is to analyse bestselling books, identify patterns in ratings, prices, reviews and genres, and experiment with Machine Learning using Linear Regression.

The project covers the complete beginner-level Data Science workflow, from dataset exploration and data quality checks to visualisation and model evaluation.

## 🎯 Objectives

- Explore the characteristics of Amazon bestselling books.
- Analyse book ratings, prices and customer reviews.
- Compare Fiction and Non-Fiction genres.
- Identify authors with the most bestseller entries.
- Understand relationships between numerical variables.
- Experiment with predicting book ratings using Linear Regression.

## 📂 Dataset

**Dataset:** Amazon Top 50 Bestselling Books (2009–2019)

The dataset contains information about bestselling books, including:

| Column | Description |
|---|---|
| Name | Title of the book |
| Author | Author's name |
| User Rating | Average customer rating |
| Reviews | Number of customer reviews |
| Price | Book price |
| Year | Year of bestseller listing |
| Genre | Fiction or Non-Fiction |

*Note: This is a historical dataset and does not represent current Amazon bestseller rankings.*

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualisation
- **Seaborn** – Statistical visualisation
- **Scikit-learn** – Machine Learning

## ⚙️ Project Workflow

### 1. Data Loading and Exploration
- Imported the required Python libraries.
- Loaded the CSV dataset using Pandas.
- Examined dataset dimensions, columns, data types and statistical summaries.

### 2. Data Quality Checks
- Checked for missing values.
- Identified duplicate rows.
- Inspected the dataset before deciding whether cleaning was necessary.

### 3. Outlier Detection
- Applied the Interquartile Range (IQR) method.
- Identified unusual values in book prices.

### 4. Data Manipulation
- Filtered highly rated books.
- Identified the 10 most-reviewed books.
- Calculated average prices and ratings by genre.
- Counted books belonging to each genre.
- Identified the top five authors by bestseller entries.

### 5. Statistical Analysis
- Calculated average ratings, prices and reviews.
- Examined correlations between numerical variables.

### 6. Data Visualisation
Created visualisations to understand patterns in the dataset:

- Distribution of book ratings.
- Number of books by genre.
- Relationship between book prices and ratings.
- Correlation heatmap.

### 7. Machine Learning – Linear Regression

Implemented a Linear Regression model to experiment with predicting book ratings.

**Features (X):**
- Price
- Reviews
- Year

**Target (y):**
- User Rating

The dataset was divided into training and testing sets using an 80:20 split.

### 8. Model Evaluation

Evaluated the model using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Also plotted actual ratings against predicted ratings to visually assess model performance.

## 📊 Visualisations

The project generates the following charts:

1. Rating Distribution
2. Books by Genre
3. Price vs User Rating
4. Correlation Heatmap
5. Actual vs Predicted Ratings

Saved visualisations can be found in the `outputs/charts/` directory.

## 📁 Project Structure

```text
amazon-best-selling-books/
│
├── data/
│   └── bestsellers with categories.csv
│
├── code/
│   └── datascience.py
│
├── outputs/
│   └── charts/
│       ├── rating_distribution.png
│       ├── books_by_genre.png
│       ├── price_vs_rating.png
│       ├── correlation_heatmap.png
│       └── actual_vs_predicted.png
│
├── README.md
└── requirements.txt
```

## 🚀 How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the script

```bash
python code/datascience.py
```

Make sure the dataset is available in the expected `data/` directory and that the CSV path in the script is configured correctly.

## 📌 Limitations

- The dataset covers historical bestseller entries from 2009 to 2019.
- The analysis does not represent current Amazon book rankings.
- Linear Regression is an introductory experiment and its predictive performance needs to be assessed against a baseline model.
- The model uses only three numerical features and does not include categorical information such as genre or author.

## 🎓 Learning Outcomes

Through this project, I gained practical experience with:

- Python libraries for Data Science.
- Data exploration and quality checks.
- Data manipulation using Pandas.
- Statistical analysis and correlation.
- Data visualisation.
- Train-test splitting.
- Implementing and evaluating a basic Machine Learning model.

## 👩‍💻 Author

A second-year Artificial Intelligence and Machine Learning engineering student exploring Data Science and Machine Learning through practical projects.
