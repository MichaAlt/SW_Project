from data_loader import load_data
from model import create_model

x_train, y_train, x_test, y_test = load_data("data_file/training_data_map5_2.csv")

# Modell erstellen
model = create_model()

# Modell trainieren
print("Modell trainieren...")
model.fit(x_train, y_train, epochs=150, batch_size=20)

# Modell evaluieren
print("\nModel evaluieren:")
loss, acc = model.evaluate(x_test, y_test)
# print("Test Accuracy: ", acc, "Test Loss: ", loss)

# Modell speichern
model.save("models_file/model5.keras")
#print("Modell gespeichert als model2.h5 ind models_file")
