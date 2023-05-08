# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Select relevant features
features = ['budget', 'genres', 'release_date', 'popularity', 'vote_average', 'vote_count']
df = df[features]

# Preprocess data
df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d')
df['year'] = df['release_date'].dt.year
df['month'] = df['release_date'].dt.month
df['day'] = df['release_date'].dt.day
df = df.drop(['release_date'], axis=1)

# Convert genres column to binary variables
genres = df['genres'].apply(lambda x: pd.Series([i['name'] for i in eval(x)]))
genres = genres.rename(columns=lambda x: 'genre_' + str(x))
df = pd.concat([df, genres], axis=1)
df = df.drop(['genres'], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['popularity'], axis=1), df['popularity'], test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)
