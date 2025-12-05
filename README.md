# INTELLIRESCUE: Intelligent Disaster Mitigation System


## Project Overview
IntelliRescue is an AI-based system designed to predict the severity of natural disasters and support emergency response planning.  
The system combines:

- Machine Learning (XGBoost) for disaster risk prediction  
- Neural Networks for deeper pattern understanding  
- Reinforcement Learning for optimizing emergency resource allocation  
- An interactive Flask web app for visualizing insights and exploring scenarios  

The goal is to improve preparedness and reduce the impact of disasters like floods, earthquakes, storms, and landslides.


## Dataset
We use the NASA **EOSDIS / EM-DAT** global disaster dataset.

- **Total records:** ~14,600  
- **Usable records after cleaning:** ~10,199  
- **Features include:**  
  - Year, Region, Country  
  - Disaster Type & Subtype  
  - Total Deaths, Injured, Affected  
  - Start/End Dates  
  - Magnitude, Damage Costs  

A disaster is classified as **high-impact** if *Total Deaths > 50*.


## Models Used

### 1. XGBoost Risk Model (Primary)
Predicts the probability that a disaster will become high-impact.  
**Performance (20% test set):**
- ROC-AUC: ~0.86–0.90  
- Accuracy: ~0.78–0.84  


### 2. Neural Network Model
A deep-learning classifier trained on one-hot encoded features (~10k dimensions).  
Used for comparison with classical ML.


### 3. Reinforcement Learning (Resource Allocation)
A Q-learning agent learns how to place limited emergency teams across regions.

- Learns from historical high-impact event frequencies  
- Produces smarter allocations than random placement  
- Outputs summaries of best regions to deploy resources  


## Web Application Features
The Flask web dashboard includes:

- **High-impact disaster prediction**  
- **Region risk analysis**  
- **Emergency resource recommendations**  
- **Interactive geographic risk map**  
- **Scenario explorer** (future years, disaster type, region)  
- **Expandable cards** for detailed statistics


Run the web app:

```bash
python3 src/app.py

