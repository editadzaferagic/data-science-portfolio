import joblib

model = joblib.load("models/product_category_model.pkl")

print("Interactive product category prediction")
print("Type product title and press Enter.")
print("Type 'exit' to quit.\n")

while True:
    text = input("Product: ").strip()

    if text.lower() == "exit":
        print("Goodbye.")
        break

    if not text:
        print("Please enter a valid product title.\n")
        continue

    prediction = model.predict([text.lower()])
    print("Category:", prediction[0], "\n")