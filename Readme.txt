#  ProbGDGT

**ProbGDGT** is a Python program that utilizes **Machine Learning (ML)** and **probability estimates from Random Forests** (Cromartie et al., 2025) to determine changes in **brGDGT provenance** within downcore sedimentary sequences.

---

##  Requirements

**Python version:** 3.x or later  

Install the required dependencies using either **pip** or **conda**:

| Package Manager | Installation Command |
|-----------------|----------------------|
| **pip** | `pip install numpy pandas matplotlib scikit-learn` |
| **conda** | `conda install numpy pandas matplotlib scikit-learn` |

---

##  Files and Directory Setup

All files required to run this program are located in the **`Code`** directory in this repository.  

You can download the entire subdirectory **`ProbGDGTs`**, which contains:
- The main script (`ProbGDGT_main_script.py`)
- The database (`ProbGDGT_SMOTE_DB.csv`)
- An example input file (`Roblesetal2022_ExampleCore_Vanevan.csv`)

---

##  Configuration

After installing the dependencies and downloading the files, update the following lines in the script with your own file paths:

```python
# Set this base directory where your input/output files are stored
# For Mac, use forward slashes (/) instead of backslashes (\)
BASE_DIR = r"C:\Users\Username\Documents\Ggdtarticle\ProbGDGTs"

# Input file names (must exist in BASE_DIR)
GDGT_FILE = "ProbGDGT_SMOTE_DB.csv"
CORES_FILE = "Roblesetal2022_ExampleCore_Vanevan.csv"

# Output directory for results (can be the same or a separate folder)
OUTPUT_DIR = r"C:\Users\Username\Documents\Ggdtarticle\ProbGDGTs\Output"

```

---

# Adjusting Bootstraps
You can modify the number of bootstraps used for confidence interval estimation.
Start with a low number (e.g., 10) during configuration to save CPU time, and increase it once the script is working correctly.

```python

# Keep the bootstrap low when testing to save CPU cycles. Increase once ready
n_bootstraps = 10

```

---

Make sure your core file is formatted similarly to the example

Once dependencies are installed and paths are configured, run the script:

```python

python ProbGDGT_main_script.py

```

___

## Output
The program will generate:
A CSV file containing probability estimates for each sedimentary environment (lake, peat, soil)
Corresponding confidence intervals
PDF graphs illustrating the results
All outputs will be saved in your specified OUTPUT_DIR.

___
## Citations
If you utilize this program, please cite:

Cromartie, A., De Jonge, C., Ménot, G., Robles, M., Dugerdil, L., Peyron, O., Rodrigo-Gámiz, M., Camuera, J., Ramos-Roman, M. J., Jiménez-Moreno, G., Colombié, C., Sahakyan, L., & Joannin, S. (2025).
Utilizing Probability Estimates from Machine Learning and Pollen to Understand the Depositional Influences on Branched GDGT in Wetlands, Peatlands, and Lakes. EGUsphere, 2025.  https://doi.org/10.5194/egusphere-2025-526

If you use the Vanevan dataset, please also cite:
Robles, M., Peyron, O., Brugiapaglia, E., Ménot, G., Dugerdil, L., Ollivier, V., Ansanay-Alex, S., Develle, A. L., Tozalakyan, P., Meliksetian, K., Sahakyan, K., Sahakyan, L., Perello, B., Badalyan, R., Colombié, C., & Joannin, S. (2022).
Impact of climate changes on vegetation and human societies during the Holocene in the South Caucasus (Vanevan, Armenia): A multiproxy approach including pollen, NPPs and brGDGTs.  Quaternary Science Reviews, 277.  https://doi.org/10.1016/j.quascirev.2021.107297


## ‍ Authors
Lead Author: A. Cromartie
Collaborators: C. De Jonge, G. Ménot, M. Robles, L. Dugerdil, O. Peyron, and co-authors
For questions, suggestions, or bug reports, please open an issue on GitHub or contact the authors.


### License
This project is distributed under the MIT License unless otherwise specified.`

