import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json, copy
import subprocess
import sys
config_select_t_data = None
config_select_model = None
folder_training = Path.cwd() / "ai" / "data_file" / "mehrklassifikation_data"
folder_models = Path.cwd() / "ai" / "models_file" 
print(folder_training)

def main():
    root = tk.Tk()

    lbg = "gray"
    root.title("AI Racing GUI")
    root.configure(background=lbg)
    root.minsize(1000, 500)
    root.maxsize(2000, 1400)
    root.geometry("300x300+50+50")
 
    tk.Label(root, text="Select a trainings data", bg=lbg).pack()
    #ttk.Combobox(root, );
    widgets = [
        tk.Label,
        ttk.Combobox,
        tk.Entry,
        tk.Button,
        tk.Radiobutton,
        tk.Scale,
        tk.Spinbox,
    ]
    data_var = tk.StringVar()
    model_var = tk.StringVar()

    config_select_t_data = ttk.Combobox(root, textvariable=data_var)
    config_select_t_data["values"] = [f.name for f in folder_training.iterdir()]
    config_select_t_data["state"] = "readonly"
    config_select_t_data.current(0)
    config_select_t_data.bind("<<ComboboxSelected>>", config_changed)
    config_select_t_data.pack(padx=5, pady=5, fill="x")
    config_select_t_data.config_key = "data"

    config_select_model = ttk.Combobox(root, textvariable=model_var)
    config_select_model["values"] = [f.name for f in folder_models.iterdir()]
    config_select_model["state"] = "readonly"
    config_select_model.current(0)
    config_select_model.bind("<<ComboboxSelected>>", config_changed)
    config_select_model.pack(padx=5, pady=5, fill="x")
    config_select_model.config_key = "model"

    text = tk.Text(root, height=1)
    text.pack(padx=5, pady=5, expand=False,fill=tk.BOTH)

    create_new_model_b = ttk.Button(root, text= "create new model",
                                     command=lambda: create_model(text))
    
    create_new_model_b.pack(padx=5, pady=5, fill="x")

    create_new_model_b = ttk.Button(root, text= "train model",
                                     command=lambda: train_model(text))
    
    create_new_model_b.pack(padx=5, pady=5, fill="x")

    root.mainloop()

def create_model(text : tk.Text):
    content = text.get("1.0", tk.END)
    print(content)
    if content is None:
        content = "new_model"
    open(Path.cwd() / "ai" / "models_file" / content / ".keras", "w").close()



def train_model(text: tk.Text):
    subprocess.run([sys.executable, Path.cwd() / "ai" / "train.py"])


def config_changed(event):
    
    config_path = Path.cwd() / "Config" / "config.json"

    # load
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print("Error loading config, using default:", e)
        return
    
    config_cpy = copy.deepcopy(config)

    if event.widget.config_key == "data":
        config["train"]["data_file"] = str(Path("data_file/mehrklassifikation_data") / event.widget.get())

    elif event.widget.config_key == "model":
        config["train"]["model_save_path"] = str(Path("models_file") / event.widget.get())

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        with open(config_path, "w") as f:
            json.dump(config_cpy, f, indent=4)

if __name__ == "__main__":
    main()
