# DataIntegration
Datasets can be found here: https://drive.google.com/drive/folders/1fw7XejeBPdim3NeV3TYEAAMMseIzFEJQ?usp=sharing

Each dataset has a swapped version where the local and external have been switched. The unswapped versions are 
amazon.zip, cord_1.zip, omdb.zip (Movie Plots), summaries_1.zip (News), wdc_1.zip, and drug_reviews.zip.

Save paths must be specified prior to running the code. This is done by creating the "output_path_config" in 
the additionalScripts directory like so,
```
processed_data|F:/data/processed_data/
database_index|F:/data/index/
experiment_results|F:/
```
Local and external data structures are stored at "processed_data" path. The indexed database is stored at 
"database_index" path. Experiment results are stored at "experiment_results" path. Set these to your preferred paths
depending on the machine. This makes it easier to run the code on different servers without manually changing paths 
every time code is pulled from git.