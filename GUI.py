import customtkinter as ctk
import model_inferece

# Create a Tkinter window
window = ctk.CTk()
window.title("LLM Model GUI")

# Set the window size
window.geometry("500x300")

# Create input fields
input_label = ctk.CTkLabel(window, text="Enter the input:")
input_label.pack()
input_entry = ctk.CTkEntry(window)
input_entry.pack()

# Create a button to trigger the prediction
predict_button = ctk.CTkButton(window, text="Predict", command=lambda: model_inferece.predict(input_entry.get()))
predict_button.pack()

# Create an output label
output_label = ctk.CTkLabel(window, text="Prediction:")
output_label.pack()

# Run the Tkinter event loop
window.mainloop()