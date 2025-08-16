import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_movies = 100
data = {
    'RottenTomatoesScore': np.random.randint(30, 100, size=num_movies),
    'SocialMediaSentiment': np.random.uniform(-1, 1, size=num_movies), # -1 negative, 1 positive
    'BoxOfficeRevenue': 1000000 + 500000 * np.random.randn(num_movies) + 100000 * np.random.randint(0,100, size=num_movies) + 50000* np.random.randint(0,100, size=num_movies)
}
df = pd.DataFrame(data)
# Ensure positive box office revenue
df['BoxOfficeRevenue'] = df['BoxOfficeRevenue'].apply(lambda x: max(0, x))
# --- 2. Data Cleaning and Preparation ---
# No significant cleaning needed for synthetic data, but this section is crucial for real-world data.
# Example: Handling missing values, outlier detection, data transformation.
# --- 3. Model Training ---
X = df[['RottenTomatoesScore', 'SocialMediaSentiment']]
y = df['BoxOfficeRevenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 4. Model Evaluation (Not shown here for brevity, but crucial in a real project) ---
# Example: R-squared, Mean Squared Error, etc.
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='RottenTomatoesScore', y='BoxOfficeRevenue', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Rotten Tomatoes Score vs. Box Office Revenue')
plt.xlabel('Rotten Tomatoes Score')
plt.ylabel('Box Office Revenue')
plt.grid(True)
plt.tight_layout()
plt.savefig('rotten_tomatoes_vs_revenue.png')
print("Plot saved to rotten_tomatoes_vs_revenue.png")
plt.figure(figsize=(10, 6))
sns.regplot(x='SocialMediaSentiment', y='BoxOfficeRevenue', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'green'})
plt.title('Social Media Sentiment vs. Box Office Revenue')
plt.xlabel('Social Media Sentiment')
plt.ylabel('Box Office Revenue')
plt.grid(True)
plt.tight_layout()
plt.savefig('sentiment_vs_revenue.png')
print("Plot saved to sentiment_vs_revenue.png")
# --- 6. Prediction (Example) ---
# new_movie = pd.DataFrame({'RottenTomatoesScore': [85], 'SocialMediaSentiment': [0.8]})
# predicted_revenue = model.predict(new_movie)
# print(f"Predicted Box Office Revenue for new movie: {predicted_revenue[0]:.2f}")