# Install prerequesites

```bash
pip install -r requirements.txt
```

# How to use our code

The notebook results.ipynb already contains most of our graphs and can be rerun at will.

In the folder models, there are script to train specific models. These can also be run directly (and an output is stored in ./models/pickels/)

# Code structure

```
.
├── welddb                        # Contains our dataset
├── dataset_analysis              # Files used to realize MNAR/MCAR/MAR analysis and PCA
├── backlog                       # File to realize PLS that was not included in the report
├── models                        # Models related repository
│   ├── pickels                   # Contains the models
│   ├── boosting_techniques.py    # XGBoost and LightGBM training file
│   ├── evaluation.py             # File containing evaluations functions
│   ├── randomforest.py.py        # Random forests training file
│   ├── regressions.py            # Linear and polynomial regressions training file
│   └── semi_supervised.py        # Semisupervised training file for XGBoost
├── format_data.py                # Data formatting related functions
├── plots.py                      # Plotting functions
├── preprocess_semi.py            # Preprocessing for the semisupervised training
├── preprocess.py                 # Preprocessing for the supervised training
├── utils.py                      # Utils functions
└── results.ipynb                 # Main notebook to train models and plot results
```