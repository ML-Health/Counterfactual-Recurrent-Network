# Copyright (c) 2020, Ioana Bica

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from CRN_encoder_evaluate import test_CRN_encoder
from CRN_decoder_evaluate import test_CRN_decoder
from utils.cancer_simulation import get_cancer_sim_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=2, type=int)
    parser.add_argument("--radio_coeff", default=2, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="crn_test_2")
    parser.add_argument("--data", default="obs",
                        help="obs, chemo, radio, chemrad, notrt") 
    parser.add_argument("--data_dir", default="/home/user/data",
                        help="Directory with pickle data files.")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=False)
    parser.add_argument("--b_decoder_hyperparm_tuning", default=False)
    parser.add_argument("--simulate", default=False,
                        help="Set to true if simulating on checkpoint model.") 
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()
    
    chemo_coeff = args.chemo_coeff
    radio_coeff = args.radio_coeff
    data_type   = args.data
    is_simulate = args.simulate
    
    if data_type == "obs": data_id = "observational_regime"
    elif data_type == "chemo": data_id = "only_chemo"
    elif data_type == "radio": data_id = "only_radio"
    elif data_type == "chemrad": data_id = "chemo_radio"
    elif data_type == "notrt": data_id = "no_tx"
    else: raise NotImplementedError(f"Data type {data_type} is not implemented!")
        
    pkl_file = os.path.join(args.data_dir, f"cancer_sim_{chemo_coeff}_{radio_coeff}_{data_id}.p")
    
    if is_simulate:
        print(f"Simulating on dataset: {pkl_file}!")
    else:
        print(f"Training & simulating on dataset: {pkl_file}!")
    
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    # Load data (rMSN DATA)
    print("Loading data...")
    
    try:
        with open(pkl_file, "rb") as f:
            pickle_map = pickle.load(f)
    except:
        raise FileNotFoundError(f"{pkl_file}")
            
    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

    models_dir = '{}/crn_models'.format(args.results_dir)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    rmse_encoder = test_CRN_encoder(pickle_map=pickle_map, models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning,
                                    is_simulate=is_simulate)


    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    """
    The counterfactual test data for a sequence of treatments in the future was simulated for a 
    projection horizon of 4 timesteps. 
   
    """

    max_projection_horizon = 4
    projection_horizon = 4
    
#     rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
    rmses_decoder, rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                                   projection_horizon=projection_horizon,
                                                   models_dir=models_dir,
                                                   encoder_model_name=encoder_model_name,
                                                   encoder_hyperparams_file=encoder_hyperparams_file,
                                                   decoder_model_name=decoder_model_name,
                                                   decoder_hyperparams_file=decoder_hyperparams_file,
                                                   b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning,
                                                   is_simulate=is_simulate)

    logging.info("Chemo coeff {} | Radio coeff {}".format(args.chemo_coeff, args.radio_coeff))
    print("RMSE for one-step-ahead prediction.")
    print(rmse_encoder)
    
    print("RMSE for 4-step-ahead prediction.")
    print(f"> Overall: {rmse_decoder}")
    print(f"> Over time: {rmses_decoder}")
    
    print("Saving RMSEs for 4-step-ahead prediction...")
    rmses_file = os.path.join(args.results_dir, f"rmses_4_step_ahead_{chemo_coeff}_{radio_coeff}_{data_id}.csv")
    np_rmses = np.array([[rmse_decoder] + rmses_decoder])
    df_rmses = pd.DataFrame(np_rmses, columns=['overall'] + [f"time {t}" for t in range(4)], index=["%RMSE"])
    df_rmses.to_csv(rmses_file)

#     print("Results for 4-step-ahead prediction (final time step only).")
#     print(rmse_decoder)
