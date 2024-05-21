# customer-targeting-for-insurance

Identify which customer (~800-1000 users) is willing to possess the insurance policy, so we campaign efficiently.

# Data and Privacy

- Due to data privacy, all of attributes (85 attributes) and ID are completely anonymized
- Attribute 1-42: Sociodemographic (All customers living in areas with the same zip code have the same socio-demographic attribute)
- Attribute 43-85: Product Ownership

# Problem Formulation

Imnbalance binary classification. The expected output is the probability of a user (or score) who is willing to buy the insurance

# Feature Overview

- Positive Correlation:
  - Customer sybtype and main type (1 & 5)
  - Household size and househould with children
  - Product ownership: Contribution to a product and corresponding amount (43-85)
- Negative Correlation:
  - Rented House vs Home owners (30 & 31)
  - National Health Services and Private Health Insurance (35 & 36)
- Potentially some features are mutually exclusive/inclusive

# Feature Selection

![](https://raw.githubusercontent.com/nvlinhvn/customer-targeting-for-insurance/main/img/entropy.png)
![](https://raw.githubusercontent.com/nvlinhvn/customer-targeting-for-insurance/main/img/feature_importance.png)
Select a useful subset of features for modeling: 47, 59, 68, 1, 5, 42, 43, 37, 18, 44

# Stacking Modeling

![](https://raw.githubusercontent.com/nvlinhvn/customer-targeting-for-insurance/main/img/stack_model.png)

# Metrics

![](https://raw.githubusercontent.com/nvlinhvn/customer-targeting-for-insurance/main/img/metrics.png)

- We see users who tends to purchase insurance are in upper-middle class with higher income, contributing significantly, and possesing high number of car/fire policies
