# Smart Home Sound Detection

This project uses a machine learning approach to detect smart-home sound events from extracted audio feature files.

The system uses an ExtraTrees classifier to predict which household sounds occur in an audio recording and estimates their onset and offset times.

The project includes a Jupyter Notebook for model development, a trained machine learning model, and a Shiny app that allows users to upload `.npz` feature files and view predicted sound events.

## Classes

The model predicts one or more of the following sound classes:

bell_ringing  
coffee_machine  
cutlery_dishes  
door_open_close  
footsteps  
keyboard_typing  
keychain  
light_switch  
microwave  
phone_ringing  
running_water  
toilet_flushing  
vacuum_cleaner  
wardrobe_drawer_open_close  
window_open_close  

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
