import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def main():

    # Prepare and load the dataset
    data = pd.read_csv("HepatitisCdata.csv", index_col=0)
    data["Sex"] = data["Sex"].map({"m": 0, "f": 1})

    # Define the classification target and features
    target = data["Category"].map({
        "0=Blood Donor": 0, "0s=suspect Blood Donor": 0,
        "1=Hepatitis": 1, "2=Fibrosis": 1, "3=Cirrhosis": 1})
    features = data.drop("Category", axis=1)

    # Split the shuffled data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)

    # Train the model using random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Use the model to make predictions on the test data
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Use the predictions on the test set to determine the model's accuracy
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Blood Donor", "Hepatitis C"]))

    # Obtain data for a new individual
    print("Data for new individual:")
    individual = pd.DataFrame([[
        float(input("Age: ")), input("Sex (m/f): "),
        float(input("ALB: ")), float(input("ALP: ")),
        float(input("ALT: ")), float(input("AST: ")),
        float(input("BIL: ")), float(input("CHE: "))]],
        columns = X_train.columns)
    
    individual["Sex"] = individual["Sex"].map({"m": 0, "f": 1})

    # Predict whether the new individual is a patient or blood donor
    prediction = model.predict(individual)[0]

    if prediction == 0:
        prediction = "Blood donor"
    else:
        prediction = "Hepatitis C patient"

    print(f"Predicted Category: {prediction}")

if __name__ == "__main__":
    main()