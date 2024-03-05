# lhmp-thor-magni-challenge-extras

<img src="assets/Logo.svg" align="left" width=25% height=25%>

## The THÖR-MAGNI Benchmark 

The THÖR-MAGNI benchmark is a motion prediction challenge, so if you work in motion 
prediction, this challenge is for you! It aims to evaluate the generalization 
capabilities of your model, so we want to evaluate it in the leave-one-scenario-out approach.
The THÖR-MAGNI dataset contains 5 scenarios that evaluate different contextual cues of human motion. 
Therefore, in this benchmark, you will need to train your model on *4* scenarios and leave the remaining scenario
for testing (validation up to you!). The observation and prediction horizons are 8 and 12 timesteps, respectively.

In the following sections, we will present the data, scripts, and examples that can help you come up with the
the brightest solution!

# Table of Contents
- [lhmp-thor-magni-challenge-extras](#lhmp-thor-magni-challenge-extras)
  - [The THÖR-MAGNI Benchmark](#the-thör-magni-benchmark)
- [Table of Contents](#table-of-contents)
  - [Install](#install)
  - [1. Data](#1-data)
  - [2. Loading Tracklets](#2-loading-tracklets)
  - [3. Examples](#3-examples)
  - [4. Submission Procedure](#4-submission-procedure)
  - [5. Terms and Conditions](#5-terms-and-conditions)
  - [Contacts](#contacts)


## Install

Install [miniconda](http://docs.conda.io/en/latest/miniconda.html). Then, you can install the required packages by running:

```
conda env create -f environment.yml
```


## 1. Data

**Data Structure (`data/` folder)**

------------
    ├── data      
    │   ├── maps     # Directory containing the obstacle maps and offsets
    │   ├── goals_positions.csv     # CSV file with the 2D positions of the 7 goal points in the environment 
    │   ├── Scenario_1.csv     # CSV file with 8-second tracklets for Scenario 1
    │   ├── Scenario_2.csv     # CSV file with 8-second tracklets for Scenario 2
    │   ├── Scenario_3.csv     # CSV file with 8-second tracklets for Scenario 3
    │   ├── Scenario_4.csv     # CSV file with 8-second tracklets for Scenario 4
    │   └── Scenario_5.csv     # CSV file with 8-second tracklets for Scenario 5
------------


The **goal_positions.csv** contains columns named as follows: 
* **day** : recording day
* **goal** : goal point identifier
* **x** : *X* coordinate for the goal point
* **y** : *Y* coordinate for the goal point


Each raw dataframe contains columns named as follows: 

* **Time** : Time in seconds (0.4s frequency)
* **frame_id** : Scene id 
* **x** : *X* coordinate	
* **y** : *Y* coordinate	
* **z** : *Z* coordinate	
* **ag_id** : human identifier	
* **data_label** : human ongoing activity
* **map_name** : file name for the map (it contains the **recording day**)
* **tracklet_id** : unique identifier for the tracklet in the corresponding scenario

Additionally, we also precompute some motion-related features such as **2D_speed** and **3D_speed**.


## 2. Loading Tracklets

To load the tracklets of a specific scneario:

```python
df = pd.read_csv(os.path.join(PATH, SCENARIO + ".csv"), index_col = "Time")
tracklets = [group for _, group in df.groupby("tracklet_id")]  # each item -> tracklet 
```

## 3. Examples

We provide scripts to help load, train, test, and visualize predictions. Specifically:
* [scaler.py](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/scaler.py) provides classes to scale features and models inputs and outputs
*  [dataloader.py](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/dataloader.py) provides classes for Pytorch Dataloaders and a "world coordinates to pixels" converter
*  [cvm.py](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/cvm.py) provides the class for the constant velocity model
* [simple_mlp.py](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/simple_mlp.py) provides Pytorch Lightning classes to train and evaluate a simple Multi-Layer Perceptron (MLP) model
* [metrics.py](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/metrics.py) provides the metrics used to evaluate your models!


To see how they work, we provide Jupyter notebooks:

* Run the Constant Velocity Model [notebook](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/run_cvm.ipynb)
* Run the Simple MLP Model [notebook](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/run_simple_mlp.ipynb).
* To get inspired on how to load the obstacle maps via PyTorch [notebook](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/run_dataloader_maps.ipynb)



## 4. Submission Procedure

Submissions to our challenge are only to be made in [**.npy** format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).

Sample jupyter notebooks for the [CVM](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/predictions_cvm.ipynb) and for the [MLP model](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras/blob/main/predictions_mlp.ipynb) demonstrate the format for predictions.

The structure of the predictions file is as follows:

[
- `predictions_scenario_1` (numpy array): Array of predicted trajectories for the first scenario.
  - `trajectory_1` (numpy array): Predicted trajectory for the first scenario.
    - `time_step_1` (numpy array): Coordinates (x, y) at time step 1.
    - `time_step_2` (numpy array): Coordinates (x, y) at time step 2.
    - ...
    - `time_step_N` (numpy array): Coordinates (x, y) at the final time step.
  - `trajectory_N` (numpy array): Predicted trajectory for the first scenario.
    - `time_step_1` (numpy array): Coordinates (x, y) at time step 1.
    - `time_step_2` (numpy array): Coordinates (x, y) at time step 2.
    - ...
    - `time_step_N` (numpy array): Coordinates (x, y) at the final time step.
- `predictions_scenario_N` (numpy array): Array of predicted trajectories for the Nth scenario.
  - ...

]

To submit, just follow the [submission procedure](https://github.com/schrtim/lhmp-thor-magni-challenge?tab=readme-ov-file#3-how-to-test-a-prediction).

## 5. Terms and Conditions

Note that the ground truth test annotations are provided along with the remaining data. This is because they match the ground truth of the original THÖR-MAGNI data, which is readily available. We trust participants to not utilize these unethically. 
Namely, we expect these train/test paradigms to be followed.


## Contacts

If you have questions or remarks regarding this challenge, please contact one of our team members:
- Benchmark evaluation: [Tim Schreiter](http://github.com/schrtim)
- Benchmark evaluation: [Janik Kaden](http://github.com/janikkaden)
- Data/Code: [Tiago Rodrigues de Almeida](http://github.com/tmralmeida)