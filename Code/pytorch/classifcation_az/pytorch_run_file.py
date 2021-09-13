from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset
from azureml.data.output_dataset_config import OutputFileDatasetConfig

#Workspace Abrufen
ws = Workspace.from_config()
#Datastore abrufen
datastore = ws.get_default_datastore()
dataset = Dataset.get_by_name(ws, name='Drohnenbilddaten')
output = OutputFileDatasetConfig(name="output").as_mount()

#Experiment Name definieren
experiment = Experiment(workspace=ws, name='BA-Experiment')

#config, definiert die source directory, u. das skript
config = ScriptRunConfig(source_directory='./src', script='model_build.py', compute_target='BA-GPU-Compute', arguments=['--data_path', dataset.as_named_input('input').as_mount(), "--output_path", output])
env = ws.environments['BA-CUDA-PYTORCH']
config.run_config.environment = env

run = experiment.submit(config)
aml_url = run.get_portal_url()
print("Submitted to compute cluster. Click link below")
print("")
print(aml_url)
