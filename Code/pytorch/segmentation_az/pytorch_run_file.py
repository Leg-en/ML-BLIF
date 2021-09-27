"""
Azure Runner File
"""


from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset
from azureml.data.output_dataset_config import OutputFileDatasetConfig

# Workspace Abrufen
ws = Workspace.from_config()
# Datastore abrufen
datastore = ws.get_default_datastore()

train_mask = Dataset.get_by_name(ws, name='Train_Masks').as_named_input("train_mask").as_mount()
train_img = Dataset.get_by_name(ws, name='Train_Images').as_named_input("train_img").as_mount()
valid_mask = Dataset.get_by_name(ws, name='Valid_Masks').as_named_input("valid_mask").as_mount()
valid_img = Dataset.get_by_name(ws, name='Valid_Images').as_named_input("valid_img").as_mount()

output = OutputFileDatasetConfig(name="output").as_mount()

# Experiment Name definieren
experiment = Experiment(workspace=ws, name='BA-Experiment')

# config, definiert die source directory, u. das skript
config = ScriptRunConfig(source_directory='./src', script='pyt_mod.py', compute_target='GPU-Compute-Bachelor',
                         arguments=['--x_train_dir', train_img, "--y_train_dir", train_mask, "--x_valid_dir", valid_img,
                                    "--y_valid_dir",valid_mask, "--output_path", output, ])
env = ws.environments['ML-CUDA-PYTORCH']
config.run_config.environment = env

run = experiment.submit(config)
aml_url = run.get_portal_url()
print("Submitted to compute cluster. Click link below")
print("")
print(aml_url)
