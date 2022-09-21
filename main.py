import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("/Users/saurabhmamgain/Downloads/delhi_temp - testset.csv")
print(df.head())
x_df = df.drop('commute_by', axis=1)
x_df = x_df.values
y_df = df['commute_by']
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, train_size=0.3)
model: DecisionTreeClassifier = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print("The accuracy score is ", score)
with open('models/model', 'wb') as f:
    pickle.dump(model, f)
with open('models/model', 'rb') as f:
    model = pickle.load(f)
    print(model.predict([[0, 308, 14]]))
