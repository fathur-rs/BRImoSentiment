from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.mongo.hooks.mongo import MongoHook

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import json
import os
from utils.helper import Helper
from dotenv import load_dotenv

from google_play_scraper import Sort, reviews
import openvino.properties.hint as hints
from openvino.runtime import Core

# Load environment variables
load_dotenv()

# Model and input details
MODEL_INT8_PATH = "./dags/models/int8/indobert.xml"
INPUT_NAMES = ["input_ids", "attention_mask"]

# Class labels for sentiment
class_labels = {
    0: "neutral",
    1: "positive",
    2: "negative"
}

# Default arguments for the DAG
default_args = {
    'owner': 'austhopia',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Helper function to scrape reviews
def scrape_reviews(app_id, start_date, end_date):
    all_reviews = []
    continuation_token = None
    
    while True:
        result, continuation_token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=1000,
            continuation_token=continuation_token
        )
        
        for review in result:
            review_date = review['at']
            if start_date <= review_date.replace(minute=0, second=0, microsecond=0) <= end_date:
                all_reviews.append(review)
            elif review_date < start_date:
                return all_reviews
        
        if not continuation_token:
            break
    
    return all_reviews

# Task function to scrape Google Playstore reviews
def google_playstore_scrapper(**kwargs):
    ti = kwargs["ti"]
    app_id = 'id.bmri.livin'
    end_datetime = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_datetime = end_datetime - timedelta(days=1)
    
    print(f"Start Scrapping: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
    reviews_list = scrape_reviews(app_id, start_datetime, end_datetime)
    
    if not reviews_list:
        print(f"No reviews found for the period {start_datetime} to {end_datetime}")
        ti.xcom_push("all_reviews", json.dumps([]))
    else:
        all_reviews_json = json.dumps(reviews_list, cls=Helper.DateTimeEncoder)
        ti.xcom_push("all_reviews", all_reviews_json)
    
    return len(reviews_list)

# Task function to check if there are any reviews
def check_reviews(**kwargs):
    ti = kwargs['ti']
    review_count = ti.xcom_pull(task_ids='google-playstore-scrapper-dag')
    if review_count == 0:
        return 'send-email-alert'
    else:
        return 'data-preprocessing-dag'

# Task function for data preprocessing
def data_preprocessing(**kwargs):
    ti = kwargs["ti"]
    all_reviews = ti.xcom_pull(task_ids='google-playstore-scrapper-dag', key='all_reviews')
    all_reviews_json = json.loads(all_reviews)
    
    df = pd.DataFrame(all_reviews_json)
    df['clean content'] = df['content'].apply(lambda x: Helper.CleanText(x))
    
    df_hf = Dataset.from_pandas(df)
    processed_df_hf = df_hf.map(Helper.preprocess_fn, batched=True)
    processed_df_list = processed_df_hf.to_dict()
    processed_df = json.dumps(processed_df_list)
    
    ti.xcom_push("transformed_reviews", processed_df)
    print(processed_df_hf.features)

# Task function for inferencing and storing results in MongoDB
def inferencing(**kwargs):
    ti = kwargs["ti"]
    processed_dataset = ti.xcom_pull(key="transformed_reviews", task_ids='data-preprocessing-dag')
    processed_dataset_json = json.loads(processed_dataset)
    processed_dataset_df = pd.DataFrame(processed_dataset_json)
    processed_dataset_hf = Dataset.from_pandas(processed_dataset_df)
    
    print(processed_dataset_hf.features)
    
    core = Core()
    config = {
        hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
        hints.num_requests: "4"
    }
    model = core.compile_model(model=MODEL_INT8_PATH, device_name="CPU", config=config)
    output_layer = model.output(0)
    
    predictions = []
    for data in tqdm(processed_dataset_hf):
        inputs = [np.expand_dims(np.asarray(data[key], dtype=np.int64), 0) for key in INPUT_NAMES]
        outputs = model(inputs)[output_layer]
        prediction = outputs[0].argmax(axis=-1)
        predictions.append(class_labels[prediction])
    
    processed_dataset_df['predicted_sentiment'] = predictions
    columns_to_exclude = ['input_ids', 'attention_mask', 'token_type_ids']
    processed_dataset_df = processed_dataset_df.drop(columns=columns_to_exclude, errors='ignore')
    processed_dataset_json = processed_dataset_df.to_json(orient='records')
    
    ti.xcom_push(key="inferenced_data", value=processed_dataset_json)

# Task function to store results in MongoDB
def store_to_mongodb(**kwargs):
    ti = kwargs["ti"]
    inferenced_data = ti.xcom_pull(task_ids='inference-dag', key='inferenced_data')
    inferenced_data_json = json.loads(inferenced_data)
    
    hook = MongoHook(conn_id='mongo_default')
    client = hook.get_conn()
    db = client[str(os.getenv('MONGODB_DB_NAME'))]
    collection = db[str(os.getenv('MONGODB_COLLECTION_NAME'))]
    
    for record in inferenced_data_json:
        at_time = record.get('at')
        if at_time:
            result = collection.update_one(
                {'at': at_time},
                {'$setOnInsert': record},
                upsert=True
            )
            if result.upserted_id:
                print(f"Inserted new document with ID: {result.upserted_id}")
            else:
                print(f"Document with 'at' = {at_time} already exists, no new document added.")
        else:
            print("No 'at' field found in review, skipping...")

# Task function to send an email alert
def send_email_alert(**kwargs):
    email = os.getenv('EMAIL')
    subject = 'Alert: Review Scrapper DAG'
    message = 'No reviews were found in the Google Playstore scraper for the past day. Please check the app and the scraper for any issues.'

    return EmailOperator(
        task_id='send-email-alert',
        to=email,
        subject=subject,
        html_content=f"<p>{message}</p>",
    ).execute(context=kwargs)

# Define the DAG
with DAG(
    'E2E_Livin_Mandiri_Review_Sentiment_Analysis_Inferencing_DAG',
    default_args=default_args,
    description='End-to-end DAG for Livin Mandiri review sentiment analysis with inferencing and MongoDB storage',
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['dev'],
) as dag:
    
    # Define tasks
    scraper_task = PythonOperator(
        task_id='google-playstore-scrapper-dag',
        python_callable=google_playstore_scrapper
    )
    
    check_reviews_task = BranchPythonOperator(
        task_id='check-reviews-dag',
        python_callable=check_reviews
    )

    transform_task = PythonOperator(
        task_id='data-preprocessing-dag',
        python_callable=data_preprocessing
    )
    
    inference_task = PythonOperator(
        task_id='inference-dag',
        python_callable=inferencing
    )
    
    store_to_mongodb_task = PythonOperator(
        task_id='store-to-mongodb-dag',
        python_callable=store_to_mongodb
    )

    email_alert_task = PythonOperator(
        task_id='send-email-alert',
        python_callable=send_email_alert
    )
    
    # Task dependencies
    scraper_task >> check_reviews_task >> [transform_task, email_alert_task]
    transform_task >> inference_task >> store_to_mongodb_task
