# Effective Entity Augmentation By Querying External Data Sources

This is the repository for the following papers:
- [Effective Entity Augmentation By Querying External Data Sources](https://dl.acm.org/doi/10.14778/3611479.3611535) (VLDB 23)
- [Generating Data Augmentation Queries Using Large Language Models](https://ceur-ws.org/Vol-3462/LLMDB3.pdf) (LLMDB 23 Workshop @ VLDB 23)

Datasets can be found here: https://drive.google.com/drive/folders/1JEUpO5CeOYwtuFuNYvA9suWPiL1nmMb8?usp=sharing

Our technical report can be found here: https://web.engr.oregonstate.edu/~termehca/papers/entityarg.pdf

Save paths must be specified prior to running the code. This is done by creating the "output_path_config" in 
the additionalScripts directory like so,
```
processed_data|F:/data/processed_data/
database_index|F:/data/index/
experiment_results|F:/
```
Local and external data structures are stored at "processed_data" path. The indexed database is stored at "database_index" path.
Experiment results are stored at "experiment_results" path. This makes it easier to run the code on different servers
without manually changing paths every time code is pulled from git.