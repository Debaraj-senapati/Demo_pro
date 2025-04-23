import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
df = pd.read_csv('symptom_data.csv')

# Label encode the target
le = LabelEncoder()
df['disease'] = le.fit_transform(df['disease'])

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Split features and target
X = df.drop('disease', axis=1)
y = df['disease']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save the model
with open('symptom_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Logistic Regression model trained and saved.")
