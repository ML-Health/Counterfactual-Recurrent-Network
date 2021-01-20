# [Estimating counterfactual treatment outcomes over time through adversarially balanced representations](https://openreview.net/forum?id=BJg866NFvB)

### Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
### Adapted by Authors of G-Net

## Dependencies

The model was implemented in Python 3.6. The following packages are needed for running the model:
- numpy==1.18.2
- pandas==1.0.4
- scipy==1.1.0
- scikit-learn==0.22.2
- tensorflow-gpu==1.15.0

## Running G-Net baseline:



    To generate a new simulation, run hyperparameter optimisation and tests for the RMSN use :

        bash test_rmsn.sh

    This produces the raw MSEs into the "results" folder, and saves calibrated models into the "models" folder

    Note that for the paper, results are imported in normalised root mean-squared error terms -> i.e. sqrt(mse) / 1150


To train and evaluate the Counterfactual Recurrent Network on tumor growth, generate tumor growth data using the rMSN baseline code. Then, run hyperparameter optimization and simulation:

```
bash run_baseline.sh
```

Before running, modify any command line arguments in the bash script if necessary. 

```bash
python test_crn.py -h
```
```
Options:

  -h, --help            show this help message and exit
  --chemo_coeff CHEMO_COEFF      # Parameter controlling the amount of time-dependent confounding applied to the chemotherapy treatment assignment.
  --radio_coeff RADIO_COEFF      # Parameter controlling the amount of time-dependent confounding applied to the radiotherapy treatment assignment. 
  --results_dir RESULTS_DIR      # Directory for saving the results.
  --model_name MODEL_NAME        # Model name
  --data DATA                    # Dataset to run on. e.g. obs, chemo, radio, chemrad, notrt
  --data_dir DATA_DIR            # Directory with pickle data files.
  --b_encoder_hyperparm_tuning   # Boolean flag for performing hyperparameter tuning for the encoder.
  --b_decoder_hyperparm_tuning   # Boolean flag for performing hyperparameter tuning for the decoder. 
  --simulate SIMULATE            # Set to true if simulating on checkpoint model.
```

Outputs:
   - Normalized root mean squared error (RMSE) for four-step ahead prediction of counterfactual outcomes (overall and over time). 
   - Trained encoder and decoder models. 

For the baseline results in the paper, hyperparameter optimization was run (this can take about 8 hours on an
NVIDIA Tesla K80 GPU). 

 
### Citation

```
@article{bica2020crn,
  title={Estimating counterfactual treatment outcomes over time through adversarially balanced representations},
  author={Bica, Ioana and Alaa, Ahmed M and Jordon, James and van der Schaar, Mihaela},
  journal={International Conference on Learning Representations},
  year={2020}
}
```
