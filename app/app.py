from shiny import App, reactive, render, ui
import numpy as np
import pandas as pd
import joblib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "extratrees_model.joblib")


# --------------------------------------------------
# Classes and features
# --------------------------------------------------

CLASS_NAMES = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]

BASE_FEATURES = [
    "zcr",
    "melspect",
    "mfcc",
    "mfcc_d",
    "mfcc_d2",
    "flux",
    "flatness",
    "centroid",
    "bandwidth",
    "contrast",
    "rolloff_low",
    "rolloff_high",
    "energy",
    "power",
]

AGGREGATIONS = ["mean", "std", "min", "max"]


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_features(npz_path):
    data = dict(np.load(npz_path, allow_pickle=True))

    feature_arrays = []

    for base in BASE_FEATURES:
        for agg in AGGREGATIONS:
            key = f"{base}_{agg}"

            if key in data:
                arr = data[key]

                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)

                feature_arrays.append(arr)

    X = np.concatenate(feature_arrays, axis=1)

    start_times = data["start_time"]
    end_times = data["end_time"]

    return X, start_times, end_times, data


def keep_full_second_segments(X, start_times, end_times):
    mask = np.isclose(start_times % 1.0, 0.0)

    return X[mask], start_times[mask], end_times[mask]


def predictions_to_intervals(preds, start_times, end_times, filename):
    rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        active = preds[:, class_idx].astype(int)

        in_event = False
        onset = None

        for i, value in enumerate(active):
            if value == 1 and not in_event:
                in_event = True
                onset = float(start_times[i])

            if value == 0 and in_event:
                offset = float(end_times[i - 1])
                rows.append({
                    "filename": filename,
                    "annotation": class_name,
                    "onset": onset,
                    "offset": offset,
                })
                in_event = False

        if in_event:
            rows.append({
                "filename": filename,
                "annotation": class_name,
                "onset": onset,
                "offset": float(end_times[-1]),
            })

    return pd.DataFrame(rows)


def run_prediction(npz_path, filename):
    X, start_times, end_times, data = load_features(npz_path)

    X, start_times, end_times = keep_full_second_segments(
        X,
        start_times,
        end_times
    )

    preds = model.predict(X)

    intervals = predictions_to_intervals(
        preds,
        start_times,
        end_times,
        filename.replace(".npz", ".wav")
    )

    return intervals, preds, start_times, end_times, data


# --------------------------------------------------
# UI
# --------------------------------------------------

app_ui = ui.page_fluid(
    ui.tags.style("""
        body {
            background: linear-gradient(135deg, #eef3ff 0%, #f8fbff 45%, #ffffff 100%);
            font-family: Arial, sans-serif;
        }

        .hero {
            background: linear-gradient(135deg, #1f3c88 0%, #2b6cb0 100%);
            color: white;
            border-radius: 22px;
            padding: 34px;
            margin-top: 22px;
            margin-bottom: 24px;
            box-shadow: 0 12px 35px rgba(31, 60, 136, 0.25);
        }

        .hero-title {
            font-size: 38px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            font-size: 17px;
            opacity: 0.92;
            max-width: 850px;
        }

        .card-box {
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 28px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.04);
        }

        .small-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 12px;
            color: #1f2937;
        }

        .info-text {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.5;
        }

        .number {
            font-size: 32px;
            font-weight: 800;
            color: #1f3c88;
        }

        .label {
            color: #6b7280;
            font-size: 14px;
        }

        table {
            font-size: 14px;
        }
    """),

    ui.div(
        ui.div("Smart Home Sound Detection", class_="hero-title"),
        ui.div(
            "Upload an extracted audio feature file and detect possible household sound events with their start and end times.",
            class_="hero-subtitle"
        ),
        class_="hero",
    ),

    ui.layout_columns(
        ui.div(
            ui.div("Upload file", class_="small-title"),
            ui.input_file(
                "file",
                "Choose a .npz feature file",
                accept=[".npz"],
                multiple=False,
            ),
            ui.p(
                "This app uses the extracted MLPC feature files. Original audio playback is only possible if .wav files are available.",
                class_="info-text",
            ),
            class_="card-box",
        ),

        ui.div(
            ui.div("Prediction summary", class_="small-title"),
            ui.output_ui("summary"),
            class_="card-box",
        ),
    ),

    ui.layout_columns(
        ui.div(
            ui.div("Detected classes", class_="small-title"),
            ui.output_table("class_table"),
            class_="card-box",
        ),

        ui.div(
            ui.div("Feature view", class_="small-title"),
            ui.output_plot("feature_plot"),
            class_="card-box",
        ),
    ),

    ui.div(
        ui.div("Predicted event intervals", class_="small-title"),
        ui.output_table("prediction_table"),
        class_="card-box",
    ),

    ui.div(
        ui.div("Sound event timeline", class_="small-title"),
        ui.output_plot("timeline_plot"),
        class_="card-box",
    ),
)


# --------------------------------------------------
# Server
# --------------------------------------------------

def server(input, output, session):

    @reactive.Calc
    def prediction_result():
        file_info = input.file()

        if file_info is None:
            return None

        file_path = file_info[0]["datapath"]
        filename = file_info[0]["name"]

        intervals, preds, start_times, end_times, data = run_prediction(
            file_path,
            filename
        )

        return {
            "intervals": intervals,
            "preds": preds,
            "start_times": start_times,
            "end_times": end_times,
            "data": data,
            "filename": filename,
        }

    @output
    @render.ui
    def summary():
        result = prediction_result()

        if result is None:
            return ui.div(
                ui.p("No file uploaded yet.", class_="info-text")
            )

        intervals = result["intervals"]
        duration = float(np.max(result["end_times"]))

        if intervals.empty:
            detected_classes = 0
        else:
            detected_classes = intervals["annotation"].nunique()

        return ui.div(
            ui.div(f"{len(intervals)}", class_="number"),
            ui.div("predicted event intervals", class_="label"),
            ui.br(),
            ui.p(f"File: {result['filename']}"),
            ui.p(f"Duration: {duration:.1f} seconds"),
            ui.p(f"Detected classes: {detected_classes}"),
        )

    @output
    @render.table
    def class_table():
        result = prediction_result()

        if result is None:
            return pd.DataFrame({"Message": ["Upload a .npz file first."]})

        intervals = result["intervals"]

        if intervals.empty:
            return pd.DataFrame({"Message": ["No sound events detected."]})

        counts = intervals["annotation"].value_counts().reset_index()
        counts.columns = ["Sound class", "Intervals"]

        return counts

    @output
    @render.table
    def prediction_table():
        result = prediction_result()

        if result is None:
            return pd.DataFrame({"Message": ["Upload a .npz file first."]})

        intervals = result["intervals"]

        if intervals.empty:
            return pd.DataFrame({"Message": ["No sound events detected."]})

        return intervals.round(2)

    @output
    @render.plot
    def feature_plot():
        result = prediction_result()

        fig, ax = plt.subplots(figsize=(8, 4))

        if result is None:
            ax.text(
                0.5,
                0.5,
                "Upload a file to show features.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

        data = result["data"]

        if "melspect_mean" in data:
            mel = data["melspect_mean"]

            ax.imshow(
                mel.T,
                aspect="auto",
                origin="lower"
            )
            ax.set_title("Mel-spectrogram feature view")
            ax.set_xlabel("Time segments")
            ax.set_ylabel("Mel bands")

        elif "energy_mean" in data:
            energy = data["energy_mean"]

            ax.plot(energy)
            ax.set_title("Energy over time")
            ax.set_xlabel("Time segments")
            ax.set_ylabel("Energy")

        else:
            ax.text(
                0.5,
                0.5,
                "No visual feature found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

        return fig

    @output
    @render.plot
    def timeline_plot():
        result = prediction_result()

        fig, ax = plt.subplots(figsize=(12, 6))

        if result is None:
            ax.text(
                0.5,
                0.5,
                "Upload a .npz file to show predictions.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

        intervals = result["intervals"]

        ax.set_title("Predicted sound events over time")
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("Sound class")

        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)

        if intervals.empty:
            ax.text(
                0.5,
                0.5,
                "No sound events detected.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        for _, row in intervals.iterrows():
            class_name = row["annotation"]

            if class_name not in CLASS_NAMES:
                continue

            y = CLASS_NAMES.index(class_name)

            ax.hlines(
                y=y,
                xmin=row["onset"],
                xmax=row["offset"],
                linewidth=6,
            )

        ax.grid(True, axis="x", alpha=0.25)

        return fig


app = App(app_ui, server)