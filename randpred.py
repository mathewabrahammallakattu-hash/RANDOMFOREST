import joblib
model = joblib.load("model.pkl")
input_data = [[6.9, 3.1, 5.4, 2.1]]
prediction = model.predict(input_data)
print(prediction)

if prediction[0] == 0:
    print("The predicted class is: Setosa,(Shortpetals, wide sepals)",",input:",input_data)
elif prediction[0] == 1:
    print("The predicted class is: Versicolor,(Medium dimensions)"",input:",input_data)
elif prediction[0] == 2:
    print("The predicted class is: Virginica,(Long petals, wide sepals)"",input:",input_data)