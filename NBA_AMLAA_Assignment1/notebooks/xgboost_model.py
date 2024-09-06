import xgboost as xgb
from sklearn.model_selection import train_test_split

# assuming y is your targets series
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

param = {
    'eta': 0.3,
    'max_depth': 3,
    'objective': 'multi:softprob',  # change this according to your problem 
    'num_class': 3}  # modify this as per the number of classes in your target variable

model = xgb.train(param, D_train, 20)

# now you can use this model to make predictions
predictions = model.predict(D_test)