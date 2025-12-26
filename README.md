Uplift Modeling Project – Promo Code Targeting

Project Overview
This project applies uplift modeling techniques to a real-world marketing problem inspired by Яндекс.Еда - a service for fast ordering and delivery of food from restaurants and groceries from stores via a mobile application or the Yandex website. 
The business objective is to increase order frequency and, as a result, grow company revenue. In this context, a conversion is defined as a successful promo code redemption. The project goal is to determine which customers should receive a 500 RUB promo code for food delivery from restaurants.

Machine Learning Objective
The ML objective is to build an uplift model from scratch using A/B test data.
The model should:
- Predict the uplift (incremental probability of promo usage) for each user;
- Satisfy a performance requirement on Uplift@30 (uplift among the top 30% of ranked users);
- Optionally log experiments and models in MLflow for reproducibility and tracking.
The final model will be used to rank users and select the top segment that maximizes incremental conversions under a fixed promo budget.

Dataset Description
The dataset contains 64,000 customers who made at least one purchase in the last 12 months and participated in an A/B test via the mobile channel. Each row is a user.
Columns:
- recency - number of months since the last purchase (engagement recency);
- history_segment - categorical segmentation of last year’s spend (customer value band);
- history - actual amount (USD) spent by the customer in the last year (monetary value);
- mens - binary flag (1/0) wheither a customer bought products for men in the last year;
- womens - binary flag (1/0) wheither a customer bought products for women in the last year;
- zip_code - encoded postal code category;
- newbie - binary flag (1/0) wheither a customer is new within the last 12 months;
- channel - main purchase channel in the last year (0 – multichannel, 1 – mobile, 2 – web);
- treatment - binary flag (1/0) wheither a customer received the promo code;
- target - biinary flag (1/0) wheither a customer successfully used the promo code.

Modeling Stack
The project uses the following main libraries:
- Python 3.10.x;
- pandas, numpy – data loading and preprocessing;
- scikit-learn – train/test split, base ML utilities;
- CatBoost, xgboost – gradient boosting models;
- scikit-uplift – uplift metrics;
- causalml – meta‑learners (BaseTLearner and BaseXLearner);
- matplotlib, seaborn, plotly – visualization (uplift curves, Qini curves, feature analysis);
- MLflow – experiment tracking and model logging.

Uplift Modeling Approaches
The following models are implemented and compared:
- CatBoostClassifier as baseline uplift model;
- T‑Learner with CatBoostClassifier - meta‑learner that trains two separate models for estimating individual treatment effects (uplift) by modeling outcomes independently in the treatment and control groups; 
- X-Learner – meta‑learner that combines four separate models for estimating individual treatment effects (uplift) by modeling outcomes in both treatment and control groups and imputing counterfactuals.
Evaluation uses uplift‑specific metrics such as Uplift@30, Qini AUC, and Uplift AUC.

Usage
1. Create virtual environment
   - install extension
   ```bash
   sudo apt-get install python3.10-venv
   ```
   - create .venv
   ```bash
   python3 -m venv .uplift-venv
   ```
   - run .venv
   ```bash
   source .uplift-venv/bin/activate
   ```
   - install packages
   ```bash
   pip install -r requirements.txt
   ```
2. Start MLFlow server
    ```bash
    bash start_mlflow_server.sh
    ```
3. Check EDA, fit uplift models, check results 
    Take a look at 
    final_project_template.ipynb
    Manually choose kernel Python(.uplift-venv)
    Run all
4. Stop MLFlow server
    go to terminal 1, press Ctrl + C
5. Deactivate virtual environment
    ```bash
    deactivate
    ```

Directory tree
.
├── catboost_info
├── data
│   └── uplift_fp_data.csv                      # Raw data
├── final_project_template.ipynb                # EDA, models trainings
├── mlruns
├── models
│   ├── baseline_catboost_model.cbm             # Fitted CatBoost baseline model
│   ├── t_learner_model.joblib                  # Fitted T-Learner
│   ├── x_learner_fe_ohe_model.joblib           # Fitted X-Learner after one-hot encoded categorical features 
                                                # and feature engineering
│   ├── x_learner_final_tuned_model.joblib      # Fitted X-Learner after one-hot encoded categorical features, 
                                                # feature engineering and hyperparameters optimization
│   ├── x_learner_initial_model.joblib          # Fitted X-Learner          
│   └── x_learner_ohe_model.joblib              # Fitted X-Learner after one-hot encoded categorical features
├── __pycache__
├── README.md
├── requirements.txt                            # Libraries to install
├── results                                     # Saved models metrics and grapths
├── sklift
├── start_mlflow_server.sh                      # MLFlow starting script
└── utils.py                                    # Uplift by percentile graph script