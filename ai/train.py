from data_loader import load_data
from model import create_model

x_train, y_train, x_test, y_test = load_data("data_file/training_data_map3.csv")

# Modell erstellen
model = create_model()

# Modell trainieren
print("Modell trainieren...")
model.fit(x_train, y_train, epochs=100, batch_size=20, shuffle = False)

# Modell evaluieren
print("Model evaluieren...")
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy: ", acc, "Test Loss: ", loss)

# Modell speichern
model.save("models_file/model3.h5")
print("Modell gespeichert als model2.h5 ind models_file")
