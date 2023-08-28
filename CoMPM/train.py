# -*- coding: utf-8 -*-
from transformers import AutoTokenizer
from ERC_dataset import *
from model import ERC_model
# from ERCcombined import ERC_model

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel, RobertaConfig 
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pdb
from tqdm import tqdm
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup

from utils import make_batch_phobert

# Cross entropy loss
def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

# Calculate accuracy
def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()

            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)

            """Calculation"""
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_labels.item()

            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list

# Save model
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    
    
## finetune Deberta-base
def main():
    """Dataset Loading"""
    batch_size = args.batch
    freeze = args.freeze
    initial = args.initial
    model_type = args.pretrained
    no_label_data = args.nolabel
    
    data_path = f"../DataPreprocess/Cleaned_Data/{no_label_data}_label/clean_data.csv"
    train_data_path = f"../DataPreprocess/Cleaned_Data/{no_label_data}_label/train_data.csv"
    dev_data_path = f"../DataPreprocess/Cleaned_Data/{no_label_data}_label/dev_data.csv"
    test_data_path = f"../DataPreprocess/Cleaned_Data/{no_label_data}_label/test_data.csv"    
    
    # Choose dataloader
    DATA_loader = Data_loader

    #Choose make batch function
    if "phobert" in model_type:
        make_batch = make_batch_phobert
    else:
        print("Pre-trained model not defined")
        return 0

    # Freeze parameters
    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'


    # Load data train/dev/test
    train_dataset = DATA_loader(train_data_path, model_type)

    dev_dataset = DATA_loader(dev_data_path, model_type)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=make_batch)

    test_dataset = DATA_loader(test_data_path, model_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=make_batch)

    # Lấy mẫu
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=make_batch)


    # Save log path
    """logging and path"""
    save_path = os.path.join(r'Model_Result')

    print(f"###Save Path### {save_path}\n", )
    log_path = os.path.join(save_path, f'{no_label_data}_final_label.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Handler
    fileHandler = logging.FileHandler(log_path)
    streamHandler = logging.StreamHandler()

    # Set logger
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)
    
    logger.info("Batch Size: %s", batch_size)
    logger.info("Freeze: %s", freeze_type)
    logger.info("Initial: %s", initial)
    logger.info("Model Type: %s", model_type)

    # Initalize model
    clsNum = len(train_dataset.labelList)
    model = ERC_model(model_type, clsNum, freeze, initial)
    model = model.cuda()
    model.train()

    """Training Setting"""
    training_epochs = args.epoch
    save_term = int(training_epochs/5)
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    optimizer = torch.optim.AdamW(model.train_params, lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) # Lập lịch điều chỉnh learning rate

    """Input & Label Setting"""
    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0
    best_epoch = 0
    for epoch in tqdm(range(1, training_epochs + 1), position=0, leave=True):
        model.train()
        for i_batch, data in enumerate(train_dataloader):

            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_labels)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        """Dev & Test evaluation"""
        model.eval()

        dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

        """Best Score & Model Save"""
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta

            test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')

            best_epoch = epoch
            logger.info(f"\nUpdate result at Epoch {best_epoch}:")
            logger.info(f'Best Dev F-score: {round(best_dev_fscore, 4)}\nDev Accuracy: {round(dev_acc, 4)}')
            logger.info(f'Test F-score: {round(test_fbeta, 4)}\nTest Accuracy: {round(test_acc, 4)}\n')
            
            _SaveModel(model, save_path)

        if epoch % 2 == 0:
            logger.info('\nEpoch: {}'.format(epoch))
            logger.info('Development ## Accuracy: {}, Precision: {}, Recall: {}, Fscore: {}\n'.format(round(dev_acc*100,2), round(dev_pre,4), round(dev_rec, 4), round(dev_fbeta,4)))
            
if __name__ == '__main__':
    torch.cuda.empty_cache()

    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)

    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5

    parser.add_argument( "--pretrained", help = 'Pretrained Model', default = f"../PhoBERT/phobert_final")
    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument('--nolabel', help = '9,8 or 7' , default = 9)
    parser.add_argument('--freeze', action='store_true', help='freezing PM')

    args, unknown = parser.parse_known_args()

    # Create Logger
    logger = logging.getLogger(__name__)

    main()