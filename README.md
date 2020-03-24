# Transprecision_Computing
Codes for IJCAI 20 and some data exploration. Below is summary of key modifications that I made to the provided codes

### 1. List of Modifications

 * Adding random seeds in any parts of codes that generate randomness (data generation, initial points of deep model,..).
 
 * Keep a log to track different metrics, e.g sum of absolute magnitude of violations, number of violated constraints.
 
 * There was a bug in model_3, that made the model_3 could not take into account of Lagrangian term in updating
 model's parameter. I corrected that. 
 
 * Revise structure of the constructor `__init__` function a bit, and add grid search opt to models.
 
 ### 2. Codes to run experiments
 
* Check `src/run_experiments.py`

