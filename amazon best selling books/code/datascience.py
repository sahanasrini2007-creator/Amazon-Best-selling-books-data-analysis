#Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
data=pd.read_csv(r"C:\Users\Lenovo\Desktop\sahana\Python for ai\MACHINE LEARNING\amazon best selling books\data\bestsellers with categories.csv")

#dataset overview
print(data.head())
print(data.tail())
print(data.shape)
print("unique elements:",data.nunique())
print(data.columns)
print(data.dtypes)
print(data.info())
print(data.describe())

#data cleaning
print("duplicate rows:",data.duplicated().sum())
print("missing values:",data.isnull().sum())


#data quality check
print("Shape:", data.shape)
print("Missing values:\n", data.isnull().sum())
print("Duplicates:", data.duplicated().sum())

#iqr method of detecting outliers
q1=data['Price'].quantile(0.25)
q3=data['Price'].quantile(0.75)
iqr=q3-q1
lower_limit=q1-(1.5*iqr)
upper_limit=q3+(1.5*iqr)
print("lower limit:",lower_limit)
print("upper limit:",upper_limit)
outliers=data[(data['Price']<lower_limit)|(data['Price']>upper_limit)]
print(outliers)

#data manipulation
highly_rated=data[data["User Rating"]>4.5]
print("highly rated book:\n")
print(highly_rated.head())

top_reviewed=data.sort_values(by='Reviews',ascending=False).head(10)
print("Top 10 most reviewed books:\n")
print(top_reviewed[['Name','Author','Reviews']])

# Average price by genre
genre_price = data.groupby('Genre')['Price'].mean()
print("\nAverage price by genre:")
print(genre_price)

# Number of books in each genre
genre_count = data.groupby('Genre').size()
print("\nBooks in each genre:")
print(genre_count)

# Average rating by genre
genre_rating = data.groupby('Genre')['User Rating'].mean()
print("\nAverage rating by genre:")
print(genre_rating)

# Top 5 authors with most bestseller entries
top_authors = data['Author'].value_counts().head(5)
print("\nTop 5 authors:")
print(top_authors)

#statistical analysis
print("\nSTATISTICAL INSIGHTS")

print("Total books:", len(data))
print("Average rating:", data['User Rating'].mean())
print("Average price:", data['Price'].mean())
print("Average reviews:", data['Reviews'].mean())

print("\nCORRELATION MATRIX:")

correlation = data[
    ['User Rating', 'Reviews', 'Price', 'Year']
].corr()

print(correlation)

#data visualization

#rating distribution
plt.figure(figsize=(10,5))
sns.histplot(data,x='User Rating',bins=10)
plt.title("distribution of ratings")
plt.xlabel("user rating")
plt.ylabel("no of books")
plt.tight_layout()
plt.show()

#books by genre
plt.figure(figsize=(10,5))
sns.countplot(data,x='Genre')
plt.title('Books by Genre')
plt.xlabel("genre")
plt.ylabel("no of books")
plt.tight_layout()
plt.show()

#price vs rating
plt.figure(figsize=(10,5))
sns.scatterplot(data, x='Price', y='User Rating', hue='Genre')
plt.title('Price vs User Rating')
plt.xlabel("price")
plt.ylabel("rating")
plt.tight_layout()
plt.show()

#correlation heatmap
plt.figure(figsize=(10,5))
sns.heatmap(correlation,annot=True)
plt.title("correlation heatmap")
plt.tight_layout()
plt.show()

#splitting of data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

X = data[['Price', 'Reviews', 'Year']]

y = data['User Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

#machine learning-linear regression
model = LinearRegression()
#fit the model
model.fit(X_train,y_train)

#predict the ratings for the test data
y_pred=model.predict(X_test)

print(y_pred)

#model evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

#actual vs predicted values
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red',linewidth=2,label="ideal prediction line")
plt.title("Linear regression: actual vs predicted values")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print("project done")