import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import datetime
import pickle
from tqdm import tqdm

max_sequence_length = 1000
root = tk.Tk()
root.withdraw()

def process_log(log_file_path, model, tokenizer):
    with open(log_file_path, 'r') as file:
        logs = file.readlines()
        first_log = logs[0]
    formatted_logs = [log.split(': ', 1)[1] for log in logs]
    sequences = tokenizer.texts_to_sequences(formatted_logs)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    predictions = model.predict(padded_sequences)
    predicted_classes = (predictions > 0.5).astype('int')

    error_indices = [i for i, prediction in enumerate(predicted_classes) if prediction == 1]
    errors = [logs[i] for i in error_indices]

    return errors

def save_errors(errors, result_file):
    with open(result_file, 'w') as result:
        for error in errors:
            result.write(error)

def process_and_save_log(log_file_path, model, tokenizer):
    errors = process_log(log_file_path, model, tokenizer)
    result_file = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_result.txt"
    save_errors(errors, result_file)
    messagebox.showinfo("Готово", "Обработка лога завершена. Результаты сохранены в файле.")

log_file_path = filedialog.askopenfilename()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('log_analysis_model.h5')

prefix_window = tk.Tk()
prefix_window.title("Префиксная строка")

first_log_label = tk.Label(prefix_window, text="Первая строка лога:")
first_log_label.pack()

with open(log_file_path, 'r') as file:
    logs = file.readlines()
    first_log = logs[0]
    first_log_text = tk.Text(prefix_window, height=1, width=100)
    first_log_text.insert(tk.END, first_log)
    first_log_text.pack()

prefix_label = tk.Label(prefix_window, text="Введите префиксную строку:")
prefix_label.pack()

prefix_entry = tk.Entry(prefix_window)
prefix_entry.pack()

def apply_prefix():
    global formatted_logs
    prefix = prefix_entry.get()
    prefix_length = len(prefix)
    formatted_logs = [log[prefix_length:] for log in logs]
    clear_logs.clear()
    for log in formatted_logs:
        clear_log = log.split(': ', 1)[1]
        clear_logs.append(clear_log)

def start():
    progress_bar = tqdm(total=1, desc="Обработка лога", unit=" лог")
    progress_bar.update(1)
    apply_prefix()
    progress_bar.total = len(formatted_logs)
    progress_bar.reset()

    process_and_save_log(log_file_path, model, tokenizer)
    progress_bar.update(1)

    progress_bar.close()

clear_logs = []
logs = []
apply_button = tk.Button(prefix_window, text="Применить", command=apply_prefix)
apply_button.pack()

start_button = tk.Button(prefix_window, text="Старт", command=start)
start_button.pack()

prefix_window.mainloop()
