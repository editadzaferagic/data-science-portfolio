# ------------------------------------------------------------
# spam_classifier.py
# Autor: Edita Džaferagić
# Opis:
# Skripta trenira jednostavan model za detekciju spam poruka.
# Sadrži: učitavanje podataka, čišćenje, vektorizaciju,
# treniranje LogisticRegression modela i interaktivno testiranje.
# ------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------
# 1) UČITAVANJE PODATAKA
# ------------------------------------------------------------

# Učitavamo CSV fajl sa porukama i kategorijama (spam/ham)
df = pd.read_csv("IMLP4_TASK_01-messages.csv")  # putanja do fajla iz zadatka

# Prikaz nekoliko redova (za provjeru)
print("\nPrvih nekoliko redova podataka:")
print(df.head())

# ------------------------------------------------------------
# 2) ANALIZA PODATAKA
# ------------------------------------------------------------

print("\nProvjera nedostajućih vrijednosti:")
print(df.isnull().sum())

print("\nVrijednosti u koloni 'category':")
print(df['category'].unique())

# ------------------------------------------------------------
# 3) ČIŠĆENJE PODATAKA
# ------------------------------------------------------------

# Uklanjamo redove koji imaju praznu kategoriju ili poruku
df = df.dropna(subset=["message", "category"])

# Pretvaramo kategorije u mala slova radi konzistentnosti
df["category"] = df["category"].str.lower()

# Standardizujemo vrijednosti – ostavljamo samo 'spam' ili 'ham'
df = df[df["category"].isin(["spam", "ham"])]

# ------------------------------------------------------------
# 4) ODVAJANJE X i y
# ------------------------------------------------------------

X = df["message"]         # ulazne poruke
y = df["category"]        # ciljne vrijednosti

# ------------------------------------------------------------
# 5) VEKTORIZACIJA – TF-IDF
# ------------------------------------------------------------

# TF-IDF pretvara tekstualne poruke u numeričke vektore
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# ------------------------------------------------------------
# 6) TRENING MODELA – Logistic Regression
# ------------------------------------------------------------

model = LogisticRegression()
model.fit(X_vectors, y)

print("\nModel uspješno istreniran!")

# ------------------------------------------------------------
# 7) INTERAKTIVNI TEST – UNOS PORUKA OD STRANE KORISNIKA
# ------------------------------------------------------------

print("\n--- INTERAKTIVNI SPAM DETEKTOR ---")
print("Unesi tekst poruke. Za izlaz upiši: exit\n")

while True:
    user_msg = input("Poruka: ")

    if user_msg.lower() == "exit":
        print("Izlaz iz programa. Hvala!")
        break

    # Pretvaranje korisničke poruke u vektor
    user_vec = vectorizer.transform([user_msg])

    # Predikcija
    prediction = model.predict(user_vec)[0]

    print(f"➡ Ova poruka je: {prediction.upper()}\n")