import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()
progress_file_path = filedialog.askopenfilename()
prefix = str(input('Введите префикс: '))
with open(progress_file_path, 'r') as file:
    logs = file.readlines()
    for stroka in logs:
        try:
            sama_stroka = stroka[len(prefix):]
            pre, nice = sama_stroka.split(': ', 1)
            with open('obucheniye.txt', 'a') as obucheniye:
                obucheniye.write(f"{nice}")
        except ValueError:
            print('ошибка валуес')