# End-to-End Autonomous Driving Model with TLCAM.

We propose an E2E model with a Traffic Light Classification Auxiliary Module (TLCAM) that emphasizes traffic signals using a fornt-view camera image.
By incorporating this auxiliary task, the model learns to focus on traffic lights in its representations, resulting in more accurate control decisions influenced by traffic light states.

---
## How to run
1. Clone this repository.
```bash
git clone git@github.com:rml-unist/e2e-driving-tlaux.git
```
2. Create a conda environment.
```bash
cd e2e-driving-tlaux
conda env create -f env.yaml
conda activate hmg
```
3. To run the training process.
The three training processes must be executed sequentially.
```bash
python3 train_1.py
```
```bash
python3 train_2.py
```
```bash
python3 train_3.py
```
**Note that** each process has to completed before starting the next one.

---

