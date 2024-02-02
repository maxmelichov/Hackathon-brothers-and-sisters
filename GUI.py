import customtkinter as ctk
import tkinter as tk
from transformers import pipeline
model_name = "distilbert-base-cased-finetuned"
classifier = pipeline('text-classification', model=model_name)

class GUI:
    def __init__(self) -> None:
        """
        Initializes the GUI class.
        """
        self.window = ctk.CTk()
        self.window.protocol("WM_DELETE_WINDOW", self.exit)
        self.create_widgets()

    def create_widgets(self):
        """
        Creates the widgets for the GUI.
        """
        # Create a Tkinter window
        self.window.title("LLM Model GUI")

        # Set the window size
        self.window.geometry("500x300")

        # Create input fields
        input_label = ctk.CTkLabel(self.window, text="Enter the input:")
        input_label.pack()
        self.input_entry = ctk.CTkEntry(self.window, width = 450, height = 200, placeholder_text="put message to dissect here", justify="left")
        self.input_entry.pack()

        # Create a button to trigger the prediction
        predict_button = ctk.CTkButton(self.window, text = "Predict", command=lambda: self.predict(self.input_entry.get()))
        predict_button.pack()

        # Create an output label
        output_label = ctk.CTkLabel(self.window, text = "Prediction:")
        output_label.pack()

        self.label = ctk.CTkLabel(self.window,text = "")
        self.label.pack()
        # Run the Tkinter event loop
        self.window.mainloop()

    def predict(self, text):
        """
        Predicts the class of the input text.
        """
        self.input_entry.delete(0,tk.END)
        prediction = classifier(text)
        if prediction[0]["label"] == "LABEL_0":
            output_text = "You are healthy"
        else:
            output_text = "Please consult a doctor"
        self.label.configure(text = output_text)

    def exit(self):
        exit()


if __name__ == "__main__":
    gui = GUI()
