import pandas as pd

df = pd.read_csv("APL_Logistics.csv",encoding='latin1')

print(df.head())

print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
# Remove unnecessary columns
df = df.drop([
    'Customer Email', 
    'Customer Fname', 
    'Customer Lname', 
    'Customer Password',
    'Customer Street',
    'Product Description'
], axis=1, errors='ignore')

print("After removing columns:")
print(df.shape)
# Convert text (categorical) to numbers
df = pd.get_dummies(df)

# Take only small sample (VERY IMPORTANT)
df = df.sample(n=20000, random_state=42)

print("After converting to numbers:")
print(df.shape)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Target column (late delivery prediction)
target = 'Late_delivery_risk'

# If column not present, try this:
if target not in df.columns:
    print("Available columns:")
    print(df.columns)
    exit()

X = df.drop(target, axis=1)
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))