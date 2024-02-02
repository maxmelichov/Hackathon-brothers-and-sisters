import customtkinter as ctk
import tkinter as tk
from transformers import pipeline
model_name = "distilbert-base-cased-finetuned"
classifier = pipeline('text-classification', model=model_name)

class GUI:
    def __init__(self) -> None:
        self.window = ctk.CTk()
        self.window.protocol("WM_DELETE_WINDOW", self.exit)
        self.create_widgets()
    def create_widgets(self):
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

    def exit(self):
        exit()


if __name__ == "__main__":
    # Add your code here
    gui = GUI()
