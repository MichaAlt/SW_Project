import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
import copy
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "Config" / "config.json"

PPO_MODELS_DIR = (
    BASE_DIR
    / "race_simulation"
    / "models_file"
    / "PPO_models_file"
)

DDPG_MODELS_DIR = (
    BASE_DIR
    / "race_simulation"
    / "models_file"
    / "DDPG_models_file"
)

SAC_MODELS_DIR = (
    BASE_DIR
    / "race_simulation"
    / "models_file"
    / "SAC_models_file"
)

MAPS_DIR = (
    BASE_DIR
    / "race_simulation"
    / "PNG_File"
)


def main():

    root = tk.Tk()

    root.title("RL Racing GUI")
    root.geometry("1000x650")
    root.minsize(1000, 500)

    root.configure(background="gray")

    frame = create_rl_frame(root)

    frame.pack(fill="both", expand=True)

    root.mainloop()


def create_rl_frame(parent):

    frame = tk.Frame(parent, bg="gray")

    tk.Label(
        frame,
        text="Reinforcement Learning",
        bg="gray",
        font=("Arial", 20)
    ).pack(pady=15)

    tk.Label(
        frame,
        text="Select RL Algorithm",
        bg="gray"
    ).pack()

    algorithm_box = ttk.Combobox(
        frame,
        state="readonly"
    )

    algorithm_box["values"] = [
        "PPO",
        "DDPG",
        "SAC"
    ]

    algorithm_box.current(0)

    update_config_algorithm(
        algorithm_box.get()
    )

    algorithm_box.pack(
        padx=10,
        pady=5,
        fill="x"
    )

    tk.Label(
        frame,
        text="Select training model",
        bg="gray"
    ).pack()

    train_model_box = ttk.Combobox(
        frame,
        state="readonly"
    )

    train_model_box["values"] = get_model_entries("PPO")

    if train_model_box["values"]:

        train_model_box.current(0)

        update_config_train_model(
            "PPO",
            train_model_box.get()
        )

    train_model_box.pack(
        padx=10,
        pady=5,
        fill="x"
    )

    tk.Label(
        frame,
        text="Select run model",
        bg="gray"
    ).pack()

    run_model_box = ttk.Combobox(
        frame,
        state="readonly"
    )

    run_model_box["values"] = get_model_entries("PPO")

    if run_model_box["values"]:

        run_model_box.current(0)

        update_config_run_model(
            "PPO",
            run_model_box.get()
        )

    run_model_box.pack(
        padx=10,
        pady=5,
        fill="x"
    )

    tk.Label(
        frame,
        text="Select map",
        bg="gray"
    ).pack()

    map_box = ttk.Combobox(
        frame,
        state="readonly"
    )

    map_box["values"] = get_folder_entries(MAPS_DIR)

    if map_box["values"]:

        map_box.current(0)

        update_config_map(
            map_box.get()
        )

    map_box.pack(
        padx=10,
        pady=5,
        fill="x"
    )

    tk.Label(
        frame,
        text="New model name",
        bg="gray"
    ).pack()

    model_name_text = tk.Text(
        frame,
        height=1
    )

    model_name_text.pack(
        padx=10,
        pady=5,
        fill="x"
    )

    ttk.Button(
        frame,
        text="Create new RL model",
        command=lambda: create_model(
            model_name_text,
            algorithm_box,
            train_model_box,
            run_model_box
        )
    ).pack(
        padx=80,
        pady=6,
        ipady=8,
        fill="x"
    )

    ttk.Button(
        frame,
        text="Train RL model",
        command=train_model
    ).pack(
        padx=80,
        pady=6,
        ipady=8,
        fill="x"
    )

    ttk.Button(
        frame,
        text="Run RL model",
        command=run_model
    ).pack(
        padx=80,
        pady=6,
        ipady=8,
        fill="x"
    )

    def algorithm_changed(event):

        algorithm = algorithm_box.get()

        values = get_model_entries(
            algorithm
        )

        train_model_box["values"] = values
        run_model_box["values"] = values

        if values:

            train_model_box.current(0)
            run_model_box.current(0)

            update_config_train_model(
                algorithm,
                train_model_box.get()
            )

            update_config_run_model(
                algorithm,
                run_model_box.get()
            )

        update_config_algorithm(
            algorithm
        )

    algorithm_box.bind(
        "<<ComboboxSelected>>",
        algorithm_changed
    )

    train_model_box.bind(
        "<<ComboboxSelected>>",
        lambda event:
        update_config_train_model(
            algorithm_box.get(),
            train_model_box.get()
        )
    )

    run_model_box.bind(
        "<<ComboboxSelected>>",
        lambda event:
        update_config_run_model(
            algorithm_box.get(),
            run_model_box.get()
        )
    )

    map_box.bind(
        "<<ComboboxSelected>>",
        lambda event:
        update_config_map(
            map_box.get()
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

    folder = get_model_dir(
        algorithm
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return sorted([
        f.name
        for f in folder.iterdir()
        if not f.name.startswith(".")
    ])


def get_folder_entries(folder):

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return sorted([
        f.name
        for f in folder.iterdir()
        if not f.name.startswith(".")
    ])


def create_model(
    text_widget,
    algorithm_box,
    train_model_box,
    run_model_box
):

    model_name = text_widget.get(
        "1.0",
        tk.END
    ).strip()

    if model_name == "":
        model_name = "new_model"

    algorithm = algorithm_box.get()

    model_dir = get_model_dir(
        algorithm
    )

    model_path = model_dir / model_name

    model_path.touch()

    values = get_model_entries(
        algorithm
    )

    train_model_box["values"] = values
    run_model_box["values"] = values

    train_model_box.set(
        model_name
    )

    run_model_box.set(
        model_name
    )

    update_config_train_model(
        algorithm,
        model_name
    )

    update_config_run_model(
        algorithm,
        model_name
    )

    print(
        f"{algorithm} model created: {model_path}"
    )


def train_model():

    train_path = (
        BASE_DIR
        / "race_simulation"
        / "train_rl.py"
    )

    subprocess.run(
        [sys.executable, str(train_path)],
        cwd=str(BASE_DIR / "race_simulation")
    )


def run_model():

    run_path = (
        BASE_DIR
        / "race_simulation"
        / "run_rl.py"
    )

    subprocess.run(
        [sys.executable, str(run_path)],
        cwd=str(BASE_DIR / "race_simulation")
    )

    subprocess.run(
        [sys.executable, str(run_path)],
        cwd=str(BASE_DIR)
    )


def load_config():

    try:

        with open(
            CONFIG_PATH,
            "r"
        ) as file:

            return json.load(file)

    except Exception as error:

        print(
            "Error loading config:",
            error
        )

        return None


def save_config(
    config,
    backup
):

    try:

        with open(
            CONFIG_PATH,
            "w"
        ) as file:

            json.dump(
                config,
                file,
                indent=4
            )

    except Exception as error:

        print(
            "Error saving config:",
            error
        )

        with open(
            CONFIG_PATH,
            "w"
        ) as file:

            json.dump(
                backup,
                file,
                indent=4
            )


def update_config_algorithm(
    algorithm
):

    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(
        config
    )

    config["train_rl"]["rl_algorithm"] = algorithm
    config["run_rl"]["rl_algorithm"] = algorithm

    save_config(
        config,
        backup
    )


def update_config_train_model(
    algorithm,
    model_name
):

    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(
        config
    )

    match algorithm:

        case "PPO":

            config["train_rl"]["model_save_path_PPO"] = (
                f"models_file/PPO_models_file/{model_name}"
            )

        case "DDPG":

            config["train_rl"]["model_save_path_DDPG"] = (
                f"models_file/DDPG_models_file/{model_name}"
            )

        case "SAC":

            config["train_rl"]["model_save_path_SAC"] = (
                f"models_file/SAC_models_file/{model_name}"
            )

    save_config(
        config,
        backup
    )


def update_config_run_model(
    algorithm,
    model_name
):

    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(
        config
    )

    match algorithm:

        case "PPO":

            config["run_rl"]["model_load_path_PPO"] = (
                f"models_file/PPO_models_file/{model_name}"
            )

        case "DDPG":

            config["run_rl"]["model_load_path_DDPG"] = (
                f"models_file/DDPG_models_file/{model_name}"
            )

        case "SAC":

            config["run_rl"]["model_load_path_SAC"] = (
                f"models_file/SAC_models_file/{model_name}"
            )

    save_config(
        config,
        backup
    )


def update_config_map(
    map_name
):

    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(
        config
    )

    map_path = str(
        Path("PNG_File") / map_name
    )

    config["train_rl"]["map_file"] = map_path
    config["run_rl"]["map_file"] = map_path

    save_config(
        config,
        backup
    )


if __name__ == "__main__":

    main()


