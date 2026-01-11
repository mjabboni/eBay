"""
eBay Auction Analysis
=====================
Business Problem: The file eBayAuctions.csv contains information on 1972 auctions 
transacted on eBay.com during May-June 2004. The goal is to use these data to build 
a model that will distinguish competitive auctions from noncompetitive ones.

A competitive auction is defined as an auction with at least two bids placed on the 
item being auctioned.
"""

# =============================================================================
# Part a: Import packages and load data
# =============================================================================
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
ebay_df = pd.read_csv('eBayAuctions.csv')

print("=" * 60)
print("eBay Auction Analysis")
print("=" * 60)
print(f"\nDataset loaded: {ebay_df.shape[0]} rows, {ebay_df.shape[1]} columns")
print("\nFirst 5 rows of the dataset:")
print(ebay_df.head())

# =============================================================================
# Part b: Observe the variables in the dataset
# =============================================================================
print("\n" + "=" * 60)
print("Part b: Variable Types and Summary Statistics")
print("=" * 60)

print("\n--- Data Types ---")
print(ebay_df.dtypes)

print("\n--- Summary Statistics ---")
print(ebay_df.describe())

print("\n--- Info about the dataset ---")
print(ebay_df.info())

# =============================================================================
# Part c: Explore categorical variables with respect to competitive auctions
# =============================================================================
print("\n" + "=" * 60)
print("Part c: Explore Categorical Variables vs Competitive Auctions")
print("=" * 60)

# Category analysis
print("\n--- Competitive Auctions by Category ---")
category_competitive = ebay_df.groupby('Category')['Competitive'].agg(['mean', 'count'])
category_competitive = category_competitive.sort_values('mean', ascending=False)
print(category_competitive)
print(f"\nCategory with MOST competitive auctions: {category_competitive['mean'].idxmax()} "
      f"({category_competitive['mean'].max():.2%} competitive rate)")

# Currency analysis
print("\n--- Competitive Auctions by Currency ---")
currency_competitive = ebay_df.groupby('currency')['Competitive'].agg(['mean', 'count'])
print(currency_competitive)

# End Day analysis
print("\n--- Competitive Auctions by End Day ---")
endday_competitive = ebay_df.groupby('endDay')['Competitive'].agg(['mean', 'count'])
endday_competitive = endday_competitive.sort_values('mean', ascending=False)
print(endday_competitive)
print(f"\nDay with MOST competitive auctions: {endday_competitive['mean'].idxmax()} "
      f"({endday_competitive['mean'].max():.2%} competitive rate)")

# Duration analysis
print("\n--- Competitive Auctions by Duration ---")
duration_competitive = ebay_df.groupby('Duration')['Competitive'].agg(['mean', 'count'])
print(duration_competitive)

# =============================================================================
# Part d: What percentage of auctions are competitive? Is there an imbalance problem?
# =============================================================================
print("\n" + "=" * 60)
print("Part d: Class Balance Analysis")
print("=" * 60)

competitive_counts = ebay_df.groupby('Competitive')['Competitive'].agg(['count'])
print("\n--- Count of Competitive vs Non-Competitive Auctions ---")
print(competitive_counts)

total = competitive_counts['count'].sum()
competitive_pct = competitive_counts.loc[1, 'count'] / total * 100
non_competitive_pct = competitive_counts.loc[0, 'count'] / total * 100

print(f"\nCompetitive auctions: {competitive_pct:.2f}%")
print(f"Non-competitive auctions: {non_competitive_pct:.2f}%")

if abs(competitive_pct - 50) > 10:
    print("\n⚠️  There appears to be a CLASS IMBALANCE issue.")
    print(f"   The classes are not evenly distributed ({competitive_pct:.1f}% vs {non_competitive_pct:.1f}%)")
else:
    print("\n✓ The classes are relatively balanced.")

# =============================================================================
# Part e: Change variable types to category for modeling
# =============================================================================
print("\n" + "=" * 60)
print("Part e: Converting Variables to Category Type")
print("=" * 60)

# Create a copy for modeling
ebay_model = ebay_df.copy()

# Convert to category type
ebay_model['currency'] = ebay_model['currency'].astype('category')
ebay_model['endDay'] = ebay_model['endDay'].astype('category')
ebay_model['Category'] = ebay_model['Category'].astype('category')
ebay_model['Competitive'] = ebay_model['Competitive'].astype('category')

print("\nVariable types after conversion:")
print(ebay_model.dtypes)

# =============================================================================
# Part f: Create dummy variables and drop reference categories
# =============================================================================
print("\n" + "=" * 60)
print("Part f: Creating Dummy Variables")
print("=" * 60)

# Create dummy variables
ebay_model = pd.get_dummies(ebay_model, prefix_sep='_', drop_first=False)

print(f"\nShape after creating dummies: {ebay_model.shape}")
print("\nColumns after creating dummies:")
print(ebay_model.columns.tolist())

# Drop reference categories (first category for each variable)
# Category_Automotive, currency_EUR, endDay_Mon
columns_to_drop = ['Category_Automotive', 'currency_EUR', 'endDay_Mon', 
                   'Competitive_0']  # Also drop one of the Competitive dummies

# Check which columns exist before dropping
existing_cols = [col for col in columns_to_drop if col in ebay_model.columns]
ebay_model.drop(columns=existing_cols, inplace=True)

# Rename Competitive_1 to Competitive
if 'Competitive_1' in ebay_model.columns:
    ebay_model.rename(columns={'Competitive_1': 'Competitive'}, inplace=True)

print(f"\nShape after dropping reference categories: {ebay_model.shape}")
print("\nFinal columns for modeling:")
print(ebay_model.columns.tolist())

# =============================================================================
# Part g: Partition data into 60% training, 40% holdout
# =============================================================================
print("\n" + "=" * 60)
print("Part g: Data Partitioning (60% Train / 40% Test)")
print("=" * 60)

# Define X (features) and y (target)
y = ebay_model['Competitive']
X = ebay_model.drop(columns=['Competitive'])

# Split the data with random_state=202
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=202
)

print(f"\nTraining set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# =============================================================================
# Modeling Part 1: Logistic Regression with ALL variables
# =============================================================================
print("\n" + "=" * 60)
print("MODELING - Part 1: Logistic Regression with ALL Variables")
print("=" * 60)

# Using statsmodels for detailed output
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit logistic regression
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit(disp=0)

print("\n--- Logistic Regression Summary (with all variables including ClosePrice) ---")
print(result.summary())

# Using sklearn for predictions
lr_model = LogisticRegression(max_iter=1000, random_state=202)
lr_model.fit(X_train, y_train)

# =============================================================================
# Modeling Part 2: Confusion Matrix and Accuracy
# =============================================================================
print("\n" + "=" * 60)
print("MODELING - Part 2: Confusion Matrix and Accuracy")
print("=" * 60)

# Predictions on test set
y_pred = lr_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix (Validation Data) ---")
print(f"                 Predicted")
print(f"                 0       1")
print(f"Actual 0      {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       1      {cm[1,0]:4d}    {cm[1,1]:4d}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# =============================================================================
# Modeling Part 3: Challenge of using ClosePrice
# =============================================================================
print("\n" + "=" * 60)
print("MODELING - Part 3: Challenge of Using ClosePrice")
print("=" * 60)

print("""
ANSWER: The challenge of using ClosePrice to predict if an auction will be 
competitive is that ClosePrice is NOT AVAILABLE at the time of prediction.

Key Issues:
1. DATA LEAKAGE: ClosePrice is determined AFTER the auction ends, which means 
   it is influenced by whether the auction was competitive or not. Using it 
   as a predictor creates data leakage - we're using information that wouldn't 
   be available when making predictions on new auctions.

2. CAUSALITY: Higher close prices often result FROM competitive bidding. 
   This reverses the causal relationship - the outcome (competitive auction) 
   causes the predictor (higher close price), not the other way around.

3. PRACTICAL UTILITY: In a real-world scenario, we want to predict whether 
   an auction WILL BE competitive before it happens. ClosePrice is only known 
   after the auction concludes, making it useless for prospective predictions.

This is why we need to build a model WITHOUT ClosePrice for practical use.
""")

# =============================================================================
# Modeling Part 4: New Model WITHOUT ClosePrice
# =============================================================================
print("\n" + "=" * 60)
print("MODELING - Part 4: Model WITHOUT ClosePrice")
print("=" * 60)

# Remove ClosePrice from features
X_no_close = X.drop(columns=['ClosePrice'])

# Split again with same random state
X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(
    X_no_close, y, test_size=0.40, random_state=202
)

# Fit new model without ClosePrice
lr_model_nc = LogisticRegression(max_iter=1000, random_state=202)
lr_model_nc.fit(X_train_nc, y_train_nc)

# Predictions
y_pred_nc = lr_model_nc.predict(X_test_nc)

# Confusion Matrix
cm_nc = confusion_matrix(y_test_nc, y_pred_nc)
print("\n--- Confusion Matrix WITHOUT ClosePrice ---")
print(f"                 Predicted")
print(f"                 0       1")
print(f"Actual 0      {cm_nc[0,0]:4d}    {cm_nc[0,1]:4d}")
print(f"       1      {cm_nc[1,0]:4d}    {cm_nc[1,1]:4d}")

# New Accuracy
accuracy_nc = accuracy_score(y_test_nc, y_pred_nc)
print(f"\nAccuracy WITHOUT ClosePrice: {accuracy_nc:.4f} ({accuracy_nc*100:.2f}%)")

# Change in accuracy
accuracy_change = accuracy_nc - accuracy
print(f"\n--- Comparison ---")
print(f"Accuracy WITH ClosePrice:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Accuracy WITHOUT ClosePrice: {accuracy_nc:.4f} ({accuracy_nc*100:.2f}%)")
print(f"Change in Accuracy:          {accuracy_change:.4f} ({accuracy_change*100:.2f}%)")

if accuracy_change < 0:
    print(f"\n⚠️  Accuracy DECREASED by {abs(accuracy_change)*100:.2f}% after removing ClosePrice.")
    print("   This confirms that ClosePrice was contributing significantly to the model,")
    print("   but for practical predictions, the model without ClosePrice is more appropriate.")
else:
    print(f"\n✓ Accuracy changed by {accuracy_change*100:.2f}% after removing ClosePrice.")

# Classification Report for model without ClosePrice
print("\n--- Classification Report (Without ClosePrice) ---")
print(classification_report(y_test_nc, y_pred_nc))

# Statsmodels summary for the model without ClosePrice
print("\n--- Logistic Regression Summary (Without ClosePrice) ---")
X_train_nc_const = sm.add_constant(X_train_nc)
logit_model_nc = sm.Logit(y_train_nc, X_train_nc_const)
result_nc = logit_model_nc.fit(disp=0)
print(result_nc.summary())

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
