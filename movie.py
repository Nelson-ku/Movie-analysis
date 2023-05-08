from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Split the data into X and y
X = df[['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']]
y = df['revenue']

# Impute missing values
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(X)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
