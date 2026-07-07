# Smart Home Sound Detection

This project uses a machine learning approach to detect smart-home sound events from extracted audio feature files.

The system uses an ExtraTrees classifier to predict which household sounds occur in an audio recording and estimates their onset and offset times.

The project includes a Jupyter Notebook for model development, a trained machine learning model, and a Shiny app that allows users to upload `.npz` feature files and view predicted sound events.

## Classes

The model predicts one or more of the following sound classes:

- bell_ringing
- coffee_machine
- cutlery_dishes
- door_open_close
- footsteps
- keyboard_typing
- keychain
- light_switch
- microwave
- phone_ringing
- running_water
- toilet_flushing
- vacuum_cleaner
- wardrobe_drawer_open_close
- window_open_close

## Project Structure

```text
smart-home-sound-detection/
│
├── app/
│   ├── app.py
│   └── model/
│       └── extratrees_model.joblib
│
├── notebooks/
│   └── challenge_solution.ipynb
│
├── assets/
│   └── plots/
│
├── docs/
│   └── MLPC_Task5_Simple_Slides.pptx
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Files

### notebooks/challenge_solution.ipynb

This notebook contains the complete model development process, including:

- loading the extracted audio feature files
- preparing the training and validation data
- reproducing the baseline model
- training ExtraTrees classifiers
- testing different hyperparameters
- evaluating the model
- testing median filtering as post-processing
- creating the final prediction CSV file

### app/app.py

This file contains the Shiny app.

The app allows the user to upload an `.npz` feature file. The file is processed in the same way as during model training, and the trained model predicts which sound events are active over time.

The app displays:

- prediction summary
- detected sound classes
- predicted onset and offset times
- a feature visualization
- a sound event timeline

### app/model/extratrees_model.joblib

This file contains the saved trained ExtraTrees model.

The Shiny app loads this file to make predictions.

### requirements.txt

This file contains the Python packages needed to run the project.

## Installation

First, download or clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/smart-home-sound-detection.git
```

Then open the project folder in CMD:

```bash
cd smart-home-sound-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Open the Notebook

Open CMD inside the project folder and start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/challenge_solution.ipynb
```

## How to Run the Shiny App

Open CMD inside the main project folder:

```bash
cd smart-home-sound-detection
```

Then run the app with:

```bash
python -m shiny run --host 127.0.0.1 --port 8010 app/app.py
```

After starting the app, open this link in your browser:

```text
http://127.0.0.1:8010
```

## How the App Works

The user uploads an `.npz` feature file.

The app loads the extracted audio features.

The features are passed to the trained ExtraTrees model.

The model predicts active sound classes for each time segment.

The segment predictions are converted into onset and offset intervals.

The detected sound events are displayed in the Shiny app.

## Model

The model is an ExtraTrees classifier trained on extracted audio features.

It was trained to detect smart-home sound events from feature files instead of raw audio files. The saved model is stored in:

```text
app/model/extratrees_model.joblib
```

## Notes

The original dataset is not included in this repository because audio datasets and extracted feature files can be large.

The repository focuses on:

- the training notebook
- the trained model
- the Shiny app
- the project structure
- the required files

The app uses `.npz` feature files. These files are not normal audio files, so they cannot be played like `.wav` files.

## Author

Luca Kasian Koren
