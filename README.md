# BRImo Review Sentiment Analysis

[![Last Commit](https://img.shields.io/github/last-commit/fathur-rs/BRImoSentiment?style=flat-square)](https://github.com/fathur-rs/BRImoSentiment/commits/main)
[![Language](https://img.shields.io/badge/language-Python-blue.svg?style=flat-square)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Orchestrator-Apache%20Airflow-brightgreen.svg?style=flat-square)](https://airflow.apache.org/)
[![OpenVINO](https://img.shields.io/badge/Inference-OpenVINO-7B2D8E.svg?style=flat-square)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-4EA94B.svg?style=flat-square)](https://www.mongodb.com/)

This project implements an end-to-end data pipeline using Apache Airflow to perform sentiment analysis on user reviews for the BRImo application from the Google Play Store. The pipeline automates the process of scraping reviews, preprocessing the text data, running inference using a pre-trained IndoBERT model (optimized with OpenVINO), and storing the results in a MongoDB database.

**Project Status**: Development (`tags=['dev']`)
**Owner**: austhopia

## Language of Use

* **Primary Language**: Python
* **Review Language Focus**: Indonesian (`lang='id'`)

## Features

* **Automated Review Scraping**: Fetches the latest reviews for the BRImo app (`id.co.bri.brimo`) from the Google Play Store.
* **Data Preprocessing**: Cleans and prepares the review text for sentiment analysis. This includes text cleaning and tokenization suitable for the IndoBERT model.
* **Sentiment Analysis**: Utilizes a fine-tuned IndoBERT model (quantized to INT8 for performance) via OpenVINO to classify reviews into neutral, positive, or negative sentiments.
* **Data Storage**: Stores the processed reviews along with their predicted sentiments into a MongoDB collection.
* **Alerting**: Sends an email notification if no new reviews are found during a scraping cycle.
* **XCom Usage**: Leverages Airflow XComs to pass data between tasks, such as the scraped reviews, preprocessed data, and inference results.

## Workflow

The DAG (`E2E_BRImo_Review_Sentiment_Analysis_Inferencing_DAG`) orchestrates the following tasks:

1.  **`google-playstore-scrapper-dag`**:
    * Scrapes reviews for the 'id.co.bri.brimo' app from the Google Play Store.
    * It fetches reviews from the last 24 hours.
    * Pushes the list of reviews (as JSON) and the count of reviews to XCom.

2.  **`check-reviews-dag`**:
    * A `BranchPythonOperator` that checks the number of reviews found by the scraper.
    * If no reviews are found (count is 0), it branches to the `send-email-alert` task.
    * Otherwise, it proceeds to the `data-preprocessing-dag` task.

3.  **`data-preprocessing-dag`**:
    * Pulls the scraped reviews from XCom.
    * Cleans the review content using a helper function (`Helper.CleanText`).
    * Transforms the data into a Hugging Face `Dataset` object.
    * Applies a preprocessing function (`Helper.preprocess_fn`) likely involving tokenization for the BERT model.
    * Pushes the processed data (as JSON) to XCom.

4.  **`inference-dag`**:
    * Pulls the preprocessed dataset from XCom.
    * Loads the INT8 quantized IndoBERT model (`./dags/models/int8/indobert.xml`) using OpenVINO Core.
    * Configures the model for throughput performance (`hints.PerformanceMode.THROUGHPUT`) and multiple requests (`hints.num_requests: "4"`).
    * Performs inference on each review to predict sentiment (neutral, positive, negative).
    * Adds the predicted sentiment to the dataset.
    * Removes unnecessary columns (`input_ids`, `attention_mask`, `token_type_ids`).
    * Pushes the inferenced data (as JSON) to XCom.

5.  **`store-to-mongodb-dag`**:
    * Pulls the inferenced data from XCom.
    * Connects to a MongoDB instance using credentials from Airflow connections (`mongo_default`) and environment variables for database and collection names.
    * Iterates through the records and upserts them into the specified MongoDB collection, using the review's 'at' timestamp as a unique identifier to avoid duplicates.

6.  **`send-email-alert`**:
    * This task is triggered if the `check-reviews-dag` finds no new reviews.
    * Sends an email notification indicating that no reviews were found. The recipient email is fetched from an environment variable.

## Technologies Used

* **Apache Airflow**: For orchestrating the data pipeline.
* **Python**: Core programming language for DAG definition and task implementation.
* **Pandas**: For data manipulation and creating DataFrames.
* **Hugging Face `datasets`**: To handle and preprocess the text data efficiently.
* **`google-play-scraper`**: Python library to scrape reviews from Google Play Store.
* **OpenVINO™ Toolkit**: For optimizing and running inference with the IndoBERT model.
* **IndoBERT**: The underlying language model for sentiment analysis (presumably pre-trained and fine-tuned for Indonesian).
* **MongoDB**: NoSQL database used to store the sentiment analysis results.
* **`dotenv`**: For managing environment variables.

## Setup and Configuration

1.  **Airflow Environment**:
    * Ensure Apache Airflow is installed and running.
    * Place the `E2E_BRImo_Review_Sentiment_Analysis_Inferencing_DAG.py` file in your Airflow DAGs folder.
    * Install all Python dependencies listed in the import section of the script (e.g., `apache-airflow`, `pandas`, `datasets`, `google-play-scraper`, `openvino`, `python-dotenv`, `pymongo`).

2.  **Model**:
    * The quantized IndoBERT model (`indobert.xml` and corresponding `.bin` file) must be present at the path specified by `MODEL_INT8_PATH` (i.e., `./dags/models/int8/indobert.xml`).

3.  **MongoDB Connection**:
    * Configure an Airflow connection named `mongo_default` with your MongoDB credentials.

4.  **Environment Variables**:
    * Create a `.env` file in your Airflow project directory or set the following environment variables in your Airflow environment:
        * `MONGODB_DB_NAME`: The name of your MongoDB database.
        * `MONGODB_COLLECTION_NAME`: The name of the collection where results will be stored.
        * `EMAIL`: The email address to send alerts to.

5.  **Helper Utilities**:
    * The script imports `Helper` from `utils.helper`. Ensure this utility file is available in the `PYTHONPATH` or in a `utils` directory accessible by the DAG (e.g., within the `dags` folder or a plugin). This helper likely contains:
        * `Helper.DateTimeEncoder`: A custom JSON encoder to handle `datetime` objects.
        * `Helper.CleanText`: A function to clean text content.
        * `Helper.preprocess_fn`: A function to tokenize and prepare data for the model.

## Running the DAG

1.  Enable the `E2E_BRImo_Review_Sentiment_Analysis_Inferencing_DAG` in the Airflow UI.
2.  The DAG is scheduled to run manually (`schedule=None`) but can be triggered as needed.
3.  Monitor the execution and check logs for any errors.
4.  Verify the sentiment analysis results in the configured MongoDB collection.

## Class Labels

The sentiment analysis model classifies reviews into the following categories:

* `0`: neutral
* `1`: positive
* `2`: negative

These labels are used to interpret the model's output.

## Contribution

* **DAG Author**: austhopia
* To contribute, please follow standard GitHub fork and pull request workflows. Ensure your changes pass any existing linting or testing setups.
