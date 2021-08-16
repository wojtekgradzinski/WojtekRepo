import joblib
model_choice = 'best_model.sav'
loaded_model = joblib.load(model_choice)
result = loaded_model.score(X_val, y_val)
print(result)