# Lung Cancer Survival Prediction API (FastAPI + Docker)

## Project Description

This project builds and deploys a Machine Learning model that predicts whether a patient will survive lung cancer. The model is exposed as a REST API using FastAPI and containerized using Docker for deployment.

---

## Machine Learning Model

### Dataset

The model is trained on a lung cancer dataset.

### Preprocessing Steps

* Dropped unnecessary columns:

  * id
  * country
  * diagnosis_date
  * end_treatment_date
* Handled missing values:

  * Filled categorical columns using **mode**
  * Filled numerical columns using **mean**
* Encoded categorical variables:

  * Gender: Male = 0, Female = 1
  * Cancer Stage:

    * Stage I = 1
    * Stage II = 2
    * Stage III = 3
    * Stage IV = 4
  * Family History: Yes = 1, No = 0
  * Smoking Status:

    * Never Smoked = 0
    * Former Smoker = 1
    * Passive Smoker = 2
    * Current Smoker = 3
  * Treatment Type:

    * Surgery = 0
    * Radiation = 1
    * Chemotherapy = 2
    * Combined = 3

### Model Used

* Logistic Regression
* The model was trained using 80% of the dataset and tested on 20%
* The trained model was saved using `joblib` as `model.joblib`

---

## FastAPI Application

The trained model is deployed using FastAPI with the following endpoints:

### GET /

Returns a message confirming the API is running.

### POST /predict

Takes patient data as input and returns prediction:

* 1 → Survived
* 0 → Did not survive

### Example Request

```json
{
  "age": 55,
  "gender": "Male",
  "bmi": 24.5,
  "cholesterol_level": 180.0,
  "hypertension": 1,
  "asthma": 0,
  "cirrhosis": 0,
  "other_cancer": 0,
  "family_history": "Yes",
  "cancer_stage": "Stage II",
  "smoking_status": "Former Smoker",
  "treatment_type": "Chemotherapy"
}
```

---

## Docker Deployment

The application is containerized using Docker.

### Build Docker Image

```bash
docker build -t lung-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 lung-api
```

---

## Access the API

* Root endpoint:
  http://127.0.0.1:8000/

* Swagger documentation:
  http://127.0.0.1:8000/docs

---

## Project Structure

```
practice6/
│
├── main.py
├── train.py
├── model.joblib
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Technologies Used

* Python
* FastAPI
* Scikit-learn
* Joblib
* NumPy
* Pandas
* Docker

---

## Conclusion

This project demonstrates the complete machine learning pipeline:

* Data preprocessing
* Model training
* Model saving
* API development
* Containerized deployment using Docker

The use of Docker ensures consistent execution across different environments.
