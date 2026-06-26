import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
import copy
import subprocess
import sys
from PIL import Image, ImageTk

# Projektwurzel
BASE_DIR = Path(__file__).resolve().parent.parent

# Pfade fuer das GUI
CONFIG_PATH = BASE_DIR / "Config" / "config.json"

CLASSIFICATION_DATA_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "data_file" / "classification_data"
REGRESSION_DATA_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "data_file" / "regression_data"
CLASSIFICATION_MODELS_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "models_file" / "classification_models"
REGRESSION_MODELS_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "models_file" / "regression_models"

REGRESSION_EXPERIMENTAL_DATA_DIR = BASE_DIR / "Supervised_Learning_Experimental" / "ai" / "data_file"
REGRESSION_EXPERIMENTAL_MODELS_DIR = BASE_DIR / "Supervised_Learning_Experimental" / "ai" / "models_file"

PPO_MODELS_DIR = BASE_DIR / "Reinforcement_Learning" / "models_file" / "PPO_models_file"
DDPG_MODELS_DIR = BASE_DIR / "Reinforcement_Learning" / "models_file" / "DDPG_models_file"
SAC_MODELS_DIR = BASE_DIR / "Reinforcement_Learning" / "models_file" / "SAC_models_file"

MAPS_DIR = BASE_DIR / "PNG_File"


def update_map_preview(map_box, preview_label, max_w=500, max_h=300):
    map_name = map_box.get()

    if not map_name:
        preview_label.config(image="", text="No map selected")
        preview_label.image = None
        return

    map_path = MAPS_DIR / map_name

    try:
        img = Image.open(map_path)
        img.thumbnail((max_w, max_h))

        tk_img = ImageTk.PhotoImage(img)

        preview_label.config(image=tk_img, text="")
        preview_label.image = tk_img

    except Exception as error:
        preview_label.config(image="", text=f"Preview error: {error}")
        preview_label.image = None


def main():
    root = tk.Tk()
    root.title("GUI")
    root.geometry("1000x720")
    root.minsize(1000, 500)
    root.configure(background="gray")

    menu_frame = tk.Frame(root, bg="#333333")
    menu_frame.pack(side="top", fill="x")

    content_frame = tk.Frame(root, bg="gray")
    content_frame.pack(side="top", fill="both", expand=True)

    classification_frame = create_model_frame(
        content_frame,
        "Classification",
        "classification",
        CLASSIFICATION_DATA_DIR,
        CLASSIFICATION_MODELS_DIR
    )

    regression_frame = create_model_frame(
        content_frame,
        "Regression",
        "regression",
        REGRESSION_DATA_DIR,
        REGRESSION_MODELS_DIR
    )

    regression_experimental_frame = create_model_frame(
        content_frame,
        "Regression Experimental",
        "regression_experimental",
        REGRESSION_EXPERIMENTAL_DATA_DIR,
        REGRESSION_EXPERIMENTAL_MODELS_DIR,
        True
    )

    rl_frame = create_rl_frame(content_frame)

    frames = {
        "classification": classification_frame,
        "regression": regression_frame,
        "regression_experimental": regression_experimental_frame,
        "reinforcement_learning": rl_frame
    }

    def show_frame(frame_name):
        for frame in frames.values():
            frame.pack_forget()

        frames[frame_name].pack(fill="both", expand=True)

    ttk.Button(menu_frame, text="Classification", command=lambda: show_frame("classification")).pack(side="left", padx=5, pady=5)
    ttk.Button(menu_frame, text="Regression", command=lambda: show_frame("regression")).pack(side="left", padx=5, pady=5)
    ttk.Button(menu_frame, text="Regression_Experimental", command=lambda: show_frame("regression_experimental")).pack(side="left", padx=5, pady=5)
    ttk.Button(menu_frame, text="Reinforcement_Learning", command=lambda: show_frame("reinforcement_learning")).pack(side="left", padx=5, pady=5)

    show_frame("classification")
    root.mainloop()


def create_model_frame(parent, title, model_type, data_folder, model_folder, experimental=False):
    frame = tk.Frame(parent, bg="gray")

    tk.Label(frame, text=f"{title} Model", bg="gray", font=("Arial", 20)).pack(pady=15)

    tk.Label(frame, text="Select training data", bg="gray").pack()
    data_box = ttk.Combobox(frame, state="readonly")
    data_box["values"] = get_folder_entries(data_folder)

    if data_box["values"]:
        data_box.current(0)
        update_config_data(model_type, data_box.get(), experimental)

    data_box.model_type = model_type
    data_box.config_key = "data"
    data_box.bind("<<ComboboxSelected>>", lambda e: config_changed(e, experimental))
    data_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select model", bg="gray").pack()
    model_box = ttk.Combobox(frame, state="readonly")
    model_box["values"] = get_folder_entries(model_folder)

    if model_box["values"]:
        model_box.current(0)
        update_config_model(model_type, model_box.get(), experimental)

    model_box.model_type = model_type
    model_box.config_key = "model"
    model_box.bind("<<ComboboxSelected>>", lambda e: config_changed(e, experimental))
    model_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select map", bg="gray").pack()
    map_box = ttk.Combobox(frame, state="readonly")
    map_box["values"] = get_folder_entries(MAPS_DIR)

    if map_box["values"]:
        map_box.current(0)
        update_config_map(map_box.get())

    map_box.config_key = "map"
    map_box.pack(padx=10, pady=5, fill="x")

    map_preview_label = tk.Label(frame, bg="gray")
    map_preview_label.pack(padx=10, pady=10)

    if map_box.get():
        update_map_preview(map_box, map_preview_label)

    map_box.bind(
        "<<ComboboxSelected>>",
        lambda e: (
            config_changed(e, experimental),
            update_map_preview(map_box, map_preview_label)
        )
    )

    tk.Label(frame, text="New training data", bg="gray").pack()
    data_name_text = tk.Text(frame, height=1)
    data_name_text.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="New model name", bg="gray").pack()
    model_name_text = tk.Text(frame, height=1)
    model_name_text.pack(padx=10, pady=5, fill="x")

    ttk.Button(frame, text=f"Create new {model_type} data file", command=lambda: create_data_file(data_name_text, model_type, data_box, data_folder, experimental)).pack(padx=80, pady=6, ipady=8, fill="x")
    ttk.Button(frame, text=f"Create new {model_type} model", command=lambda: create_model(model_name_text, model_type, model_box, model_folder, experimental)).pack(padx=80, pady=6, ipady=8, fill="x")
    ttk.Button(frame, text=f"Start {model_type} data collection", command=lambda: start_simulation("data_collection", model_type, model_box, map_box, experimental)).pack(padx=80, pady=6, ipady=8, fill="x")
    ttk.Button(frame, text=f"Train {model_type} model", command=lambda: train_model(model_type, model_box, experimental)).pack(padx=80, pady=6, ipady=8, fill="x")
    ttk.Button(frame, text=f"Run {model_type} model", command=lambda: start_simulation("ai_run", model_type, model_box, map_box, experimental)).pack(padx=80, pady=6, ipady=8, fill="x")

    tk.Label(frame, text="Select optimizer", bg="gray").pack()
    optimizer_box = ttk.Combobox(frame, state="readonly")
    optimizer_box["values"] = ["adam", "sgd"]
    optimizer_box.current(0)

    update_config_optimizer(optimizer_box.get())

    optimizer_box.config_key = "optimizer"
    optimizer_box.bind("<<ComboboxSelected>>", lambda e: config_changed(e, experimental))
    optimizer_box.pack(padx=10, pady=5, fill="x")

    return frame


def get_folder_entries(folder):
    folder.mkdir(parents=True, exist_ok=True)

    return sorted([
        f.name
        for f in folder.iterdir()
        if not f.name.startswith(".")
    ])


def create_model(text_widget, model_type, model_box, model_folder, experimental):
    model_name = text_widget.get("1.0", tk.END).strip()

    if model_name == "":
        model_name = "new_model"

    if not model_name.endswith(".keras"):
        model_name += ".keras"

    model_path = model_folder / model_name
    model_path.touch()

    model_box["values"] = get_folder_entries(model_folder)
    model_box.set(model_name)

    update_prediction_type(model_type)
    update_config_model(model_type, model_name, experimental)

    print(f"{model_type} model created: {model_path}")


def create_data_file(text_widget, model_type, data_box, data_folder, experimental=False):
    file_name = text_widget.get("1.0", tk.END).strip()

    if file_name == "":
        file_name = "new_data"

    if not file_name.endswith(".csv"):
        file_name += ".csv"

    file_path = data_folder / file_name
    file_path.touch()

    data_box["values"] = get_folder_entries(data_folder)
    data_box.set(file_name)

    update_config_data(model_type, file_name, experimental)

    print(f"{model_type} data file created: {file_path}")


def start_simulation(mode, model_type, model_box, map_box, experimental):
    update_mode(mode)
    update_prediction_type(model_type)

    if model_box.get():
        update_config_model(model_type, model_box.get(), experimental)

    if map_box.get():
        update_config_map(map_box.get())

    if experimental:
        if mode == "data_collection":
            manual_run(True)
        else:
            run_model("regression", model_box, map_box, experimental)
        return

    simulation_path = BASE_DIR / "Supervised_Learning" / "race_simulation" / "simulation.py"
    simulation_dir = BASE_DIR / "Supervised_Learning" / "race_simulation"

    subprocess.run(
        [sys.executable, str(simulation_path)],
        cwd=str(simulation_dir)
    )


def train_model(model_type, model_box, experimental):
    update_prediction_type(model_type)

    if model_box.get():
        update_config_model(model_type, model_box.get(), experimental)

    if experimental:
        train_path = BASE_DIR / "Supervised_Learning_Experimental" / "ai" / "train_exp.py"
        working_dir = BASE_DIR / "Supervised_Learning_Experimental"
    else:
        train_path = BASE_DIR / "Supervised_Learning" / "ai" / "train.py"
        working_dir = BASE_DIR / "Supervised_Learning"

    subprocess.run(
        [sys.executable, str(train_path)],
        cwd=str(working_dir)
    )


def update_mode(mode):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)
    config["mode"]["mode"] = mode
    save_config(config, backup)


def update_config_optimizer(optimizer_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("model", {})
    config["model"]["optimizer"] = optimizer_name

    save_config(config, backup)


def manual_run(experimental):
    if experimental is False:
        manual_run_path = BASE_DIR / "Supervised_Learning" / "race_simulation" / "manual_run.py"
        race_simulation_dir = BASE_DIR / "Supervised_Learning" / "race_simulation"
    else:
        manual_run_path = BASE_DIR / "Supervised_Learning_Experimental" / "race_simulation" / "manual_run_exp.py"
        race_simulation_dir = BASE_DIR / "Supervised_Learning_Experimental" / "race_simulation"

    subprocess.run(
        [sys.executable, str(manual_run_path)],
        cwd=str(race_simulation_dir)
    )


def run_model(model_type, model_box, map_box, experimental):
    update_prediction_type(model_type)

    if model_box.get():
        update_config_model(model_type, model_box.get(), experimental)

    if map_box.get():
        update_config_map(map_box.get())

    if experimental is False:
        ai_run_path = BASE_DIR / "Supervised_Learning" / "race_simulation" / "ai_run.py"
        race_simulation_dir = BASE_DIR / "Supervised_Learning" / "race_simulation"
    else:
        ai_run_path = BASE_DIR / "Supervised_Learning_Experimental" / "race_simulation" / "ai_run_exp.py"
        race_simulation_dir = BASE_DIR / "Supervised_Learning_Experimental" / "race_simulation"

    if not ai_run_path.exists():
        print(f"ai_run.py not found: {ai_run_path}")
        return

    print(f"Running {model_type} model...")

    subprocess.run(
        [sys.executable, str(ai_run_path)],
        cwd=str(race_simulation_dir)
    )


def config_changed(event, experimental):
    widget = event.widget

    if widget.config_key == "data":
        update_config_data(widget.model_type, widget.get(), experimental)

    elif widget.config_key == "model":
        update_config_model(widget.model_type, widget.get(), experimental)

    elif widget.config_key == "map":
        update_config_map(widget.get())

    elif widget.config_key == "optimizer":
        update_config_optimizer(widget.get())


def load_config():
    try:
        with open(CONFIG_PATH, "r") as file:
            return json.load(file)
    except Exception as error:
        print("Error loading config:", error)
        return None


def save_config(config, backup):
    try:
        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)
    except Exception as error:
        print("Error saving config:", error)

        with open(CONFIG_PATH, "w") as file:
            json.dump(backup, file, indent=4)


def update_prediction_type(model_type):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("prediction", {})
    config["prediction"]["prediction"] = model_type

    save_config(config, backup)


def update_config_data(model_type, file_name, experimental=False):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("train", {})
    config.setdefault("simulation", {})
    config.setdefault("train_regression_exp", {})
    config.setdefault("manual_run_exp", {})
    config.setdefault("auto_collect_centerline_exp", {})

    if model_type == "classification":
        config["train"]["data_file_classification"] = str(Path("data_file") / "classification_data" / file_name)
        config["simulation"]["data_save_path_classification"] = str(Path("..") / "ai" / "data_file" / "classification_data" / file_name)

    elif model_type == "regression":
        if not experimental:
            config["train"]["data_file_regression"] = str(Path("data_file") / "regression_data" / file_name)
            config["simulation"]["data_save_path_regression"] = str(Path("..") / "ai" / "data_file" / "regression_data" / file_name)
        else:
            config["train_regression_exp"]["data_file"] = str(Path("ai") / "data_file" / file_name)
            save_path = str(Path("..") / "ai" / "data_file" / file_name)
            config["manual_run_exp"]["data_save_path_regression"] = save_path
            config["auto_collect_centerline_exp"]["data_save_path"] = save_path

    save_config(config, backup)


def update_config_model(model_type, model_name, experimental=False):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("train", {})
    config.setdefault("ai_run", {})

    if model_type == "classification":
        train_path = Path("models_file") / "classification_models" / model_name
        run_path = Path("..") / "ai" / "models_file" / "classification_models" / model_name

        config["train"]["model_save_path_classification"] = str(train_path)
        config["ai_run"]["model_path_classification"] = str(run_path)

    elif model_type == "regression":
        if experimental is False:
            train_path = Path("models_file") / "regression_models" / model_name
            run_path = Path("..") / "ai" / "models_file" / "regression_models" / model_name

            config["train"]["model_save_path_regression"] = str(train_path)
            config["ai_run"]["model_path_regression"] = str(run_path)
        else:
            config.setdefault("train_regression_exp", {})
            config.setdefault("ai_run_exp", {})

            train_path = Path("models_file") / model_name
            run_path = Path("..") / "ai" / "models_file" / model_name

            config["train_regression_exp"]["model_save_path"] = str(train_path)
            config["ai_run_exp"]["model_path"] = str(run_path)

    save_config(config, backup)


def update_config_map(map_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("simulation", {})
    config.setdefault("ai_run", {})

    map_path = str(Path("../../PNG_File") / map_name)

    config["simulation"]["map_file"] = map_path

    save_config(config, backup)


def create_rl_frame(parent):
    frame = tk.Frame(parent, bg="gray")

    tk.Label(frame, text="Reinforcement Learning", bg="gray", font=("Arial", 20)).pack(pady=15)

    tk.Label(frame, text="Select RL Algorithm", bg="gray").pack()
    algorithm_box = ttk.Combobox(frame, state="readonly")
    algorithm_box["values"] = ["PPO", "DDPG", "SAC"]
    algorithm_box.current(0)
    update_config_algorithm_rl(algorithm_box.get())
    algorithm_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select training model", bg="gray").pack()
    train_model_box = ttk.Combobox(frame, state="readonly")
    train_model_box["values"] = get_model_entries("PPO")

    if train_model_box["values"]:
        train_model_box.current(0)
        update_config_train_model_rl("PPO", train_model_box.get())

    train_model_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select run model", bg="gray").pack()
    run_model_box = ttk.Combobox(frame, state="readonly")
    run_model_box["values"] = get_model_entries("PPO")

    if run_model_box["values"]:
        run_model_box.current(0)
        update_config_run_model_rl("PPO", run_model_box.get())

    run_model_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select map", bg="gray").pack()
    map_box = ttk.Combobox(frame, state="readonly")
    map_box["values"] = get_folder_entries(MAPS_DIR)

    if map_box["values"]:
        map_box.current(0)
        update_config_map_rl(map_box.get())

    map_box.pack(padx=10, pady=5, fill="x")

    map_preview_label = tk.Label(frame, bg="gray")
    map_preview_label.pack(padx=10, pady=10)

    if map_box.get():
        update_map_preview(map_box, map_preview_label)

    tk.Label(frame, text="New model name", bg="gray").pack()
    model_name_text = tk.Text(frame, height=1)
    model_name_text.pack(padx=10, pady=5, fill="x")

    ttk.Button(frame, text="Create new RL model", command=lambda: create_model_rl(model_name_text, algorithm_box, train_model_box, run_model_box)).pack(padx=80, pady=6, ipady=8, fill="x")
    ttk.Button(frame, text="Train RL model", command=train_model_rl).pack(padx=80, pady=6, ipady=8, fill="x")
    ttk.Button(frame, text="Run RL model", command=run_model_rl).pack(padx=80, pady=6, ipady=8, fill="x")

    def algorithm_changed(event):
        algorithm = algorithm_box.get()
        values = get_model_entries(algorithm)

        train_model_box["values"] = values
        run_model_box["values"] = values

        if values:
            train_model_box.current(0)
            run_model_box.current(0)
            update_config_train_model_rl(algorithm, train_model_box.get())
            update_config_run_model_rl(algorithm, run_model_box.get())

        update_config_algorithm_rl(algorithm)

    algorithm_box.bind("<<ComboboxSelected>>", algorithm_changed)

    train_model_box.bind(
        "<<ComboboxSelected>>",
        lambda event: update_config_train_model_rl(
            algorithm_box.get(),
            train_model_box.get()
        )
    )

    run_model_box.bind(
        "<<ComboboxSelected>>",
        lambda event: update_config_run_model_rl(
            algorithm_box.get(),
            run_model_box.get()
        )
    )

    map_box.bind(
        "<<ComboboxSelected>>",
        lambda event: (
            update_config_map_rl(map_box.get()),
            update_map_preview(map_box, map_preview_label)
        )
    )

    return frame


def get_model_dir(algorithm):
    match algorithm:
        case "PPO":
            return PPO_MODELS_DIR
        case "DDPG":
            return DDPG_MODELS_DIR
        case "SAC":
            return SAC_MODELS_DIR


def get_model_entries(algorithm):
    folder = get_model_dir(algorithm)
    folder.mkdir(parents=True, exist_ok=True)

    return sorted([
        f.name
        for f in folder.iterdir()
        if not f.name.startswith(".")
    ])


def create_model_rl(text_widget, algorithm_box, train_model_box, run_model_box):
    model_name = text_widget.get("1.0", tk.END).strip()

    if model_name == "":
        model_name = "new_model"

    if not model_name.endswith(".zip"):
        model_name += ".zip"

    algorithm = algorithm_box.get()
    model_dir = get_model_dir(algorithm)
    model_path = model_dir / model_name
    model_path.touch()

    values = get_model_entries(algorithm)

    train_model_box["values"] = values
    run_model_box["values"] = values

    train_model_box.set(model_name)
    run_model_box.set(model_name)

    update_config_train_model_rl(algorithm, model_name)
    update_config_run_model_rl(algorithm, model_name)

    print(f"{algorithm} model created: {model_path}")


def train_model_rl():
    train_path = BASE_DIR / "Reinforcement_Learning" / "train_rl.py"

    subprocess.run(
        [sys.executable, str(train_path)],
        cwd=str(BASE_DIR / "Reinforcement_Learning")
    )


def run_model_rl():
    run_path = BASE_DIR / "Reinforcement_Learning" / "run_rl.py"

    subprocess.run(
        [sys.executable, str(run_path)],
        cwd=str(BASE_DIR / "Reinforcement_Learning")
    )


def update_config_algorithm_rl(algorithm):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("train_rl", {})
    config.setdefault("run_rl", {})

    config["train_rl"]["rl_algorithm"] = algorithm
    config["run_rl"]["rl_algorithm"] = algorithm

    save_config(config, backup)


def update_config_train_model_rl(algorithm, model_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("train_rl", {})

    match algorithm:
        case "PPO":
            config["train_rl"]["model_save_path_PPO"] = f"models_file/PPO_models_file/{model_name}"
        case "DDPG":
            config["train_rl"]["model_save_path_DDPG"] = f"models_file/DDPG_models_file/{model_name}"
        case "SAC":
            config["train_rl"]["model_save_path_SAC"] = f"models_file/SAC_models_file/{model_name}"

    save_config(config, backup)


def update_config_run_model_rl(algorithm, model_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("run_rl", {})

    match algorithm:
        case "PPO":
            config["run_rl"]["model_load_path_PPO"] = f"models_file/PPO_models_file/{model_name}"
        case "DDPG":
            config["run_rl"]["model_load_path_DDPG"] = f"models_file/DDPG_models_file/{model_name}"
        case "SAC":
            config["run_rl"]["model_load_path_SAC"] = f"models_file/SAC_models_file/{model_name}"

    save_config(config, backup)


def update_config_map_rl(map_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("train_rl", {})
    config.setdefault("run_rl", {})

    map_path = str(Path("../PNG_File") / map_name)

    config["train_rl"]["map_file"] = map_path
    config["run_rl"]["map_file"] = map_path

    save_config(config, backup)


if __name__ == "__main__":
    main()