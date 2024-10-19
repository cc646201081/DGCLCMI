from parser import parse_args

import torch
import torch.nn as nn
import torch.optim as optim

from NGCF import NGCF

from dataloader import data_generator

import warnings

warnings.filterwarnings('ignore')
from time import time

import numpy as np
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, matthews_corrcoef, accuracy_score, \
    roc_auc_score, average_precision_score

import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return


def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 1
        while counter < len(row):
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):
    SampleFeature = []

    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        FeatureMatrix = []
        counter1 = 0
        while counter1 < PaddingLength:
            row = []
            counter2 = 0
            while counter2 < len(EmbeddingList[0]) - 1:
                row.append(0)
                counter2 = counter2 + 1
            FeatureMatrix.append(row)
            counter1 = counter1 + 1

        try:
            counter3 = 0
            while counter3 < PaddingLength:
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
                    counter4 = counter4 + 1
                counter3 = counter3 + 1
        except:
            pass

        PairFeature.append(FeatureMatrix)
        SampleFeature.append(PairFeature)
        counter = counter + 1
    return SampleFeature


def model_train(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = []
    total_specificity = []
    total_precision = []
    total_sensitivity = []
    total_mcc = []
    total_accuracy = []
    total_roc_auc = []
    total_aupr = []

    for miRNA, circRNA, y_true in train_loader:
        u_embeddings, i_embeddings = model(miRNA, circRNA)
        y_scores = torch.mm(u_embeddings, i_embeddings.T).diag()
        loss = criterion(y_scores, y_true.to("cuda:0"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_scores = y_scores.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred = np.where(y_scores >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        precision = precision_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        total_loss.append(loss.item())
        total_specificity.append(specificity)
        total_precision.append(precision)
        total_sensitivity.append(sensitivity)
        total_mcc.append(mcc)
        total_accuracy.append(accuracy)
        total_roc_auc.append(roc_auc)
        total_aupr.append(aupr)
        # print(f"loss: {loss}")
    total_loss = np.array(total_loss).mean()
    total_specificity = np.array(total_specificity).mean()
    total_precision = np.array(total_precision).mean()
    total_sensitivity = np.array(total_sensitivity).mean()
    total_mcc = np.array(total_mcc).mean()
    total_accuracy = np.array(total_accuracy).mean()
    total_roc_auc = np.array(total_roc_auc).mean()
    total_aupr = np.array(total_aupr).mean()

    return total_loss, total_specificity, total_precision, total_sensitivity, total_mcc, total_accuracy, total_roc_auc, total_aupr


def model_test(model, criterion, test_loader):
    model.eval()
    total_loss = []
    total_specificity = []
    total_precision = []
    total_sensitivity = []
    total_mcc = []
    total_accuracy = []
    total_roc_auc = []
    total_aupr = []

    for miRNA, circRNA, y_true in test_loader:
        u_embeddings, i_embeddings = model(miRNA, circRNA)
        # loss = model.create_bpr_loss(u_embeddings, i_embeddings)  # , batch_mf_loss, batch_emb_loss

        y_scores = torch.mm(u_embeddings, i_embeddings.T).diag()
        loss = criterion(y_scores, y_true.to("cuda:0"))

        y_scores = y_scores.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        y_pred = np.where(y_scores >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        precision = precision_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        total_loss.append(loss.item())
        total_specificity.append(specificity)
        total_precision.append(precision)
        total_sensitivity.append(sensitivity)
        total_mcc.append(mcc)
        total_accuracy.append(accuracy)
        total_roc_auc.append(roc_auc)
        total_aupr.append(aupr)

    total_loss = np.array(total_loss).mean()
    total_specificity = np.array(total_specificity).mean()
    total_precision = np.array(total_precision).mean()
    total_sensitivity = np.array(total_sensitivity).mean()
    total_mcc = np.array(total_mcc).mean()
    total_accuracy = np.array(total_accuracy).mean()
    total_roc_auc = np.array(total_roc_auc).mean()
    total_aupr = np.array(total_aupr).mean()

    return total_loss, total_specificity, total_precision, total_sensitivity, total_mcc, total_accuracy, total_roc_auc,total_aupr



if __name__ == '__main__':


    np.random.seed(0)
    torch.manual_seed(0)
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id))
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    for data_name in [ 'CMI-9589','CMI-9905','CMI-20208']:
        args.dataset = data_name
        for CounterT in range(5):
            train_loader, test_loader, norm_adj, n_users, n_items = data_generator(
                f'Dataset/{args.dataset}/Positive_Sample_Train' + str(CounterT) + '.csv',
                f'Dataset/{args.dataset}/Negative_Sample_Train' + str(CounterT) + '.csv',
                f'Dataset/{args.dataset}/Positive_Sample_Test' + str(CounterT) + '.csv',
                f'Dataset/{args.dataset}/Negative_Sample_Test' + str(CounterT) + '.csv', args)

            AllMiSequence = []
            ReadMyCsv1(AllMiSequence, f'Dataset/{args.dataset}/miRNA_order.csv')  # 加载所有miRNA
            miRNAEmbedding = []
            ReadMyCsv3(miRNAEmbedding, f'Dataset/{args.dataset}/miRNAEmbedding.csv')  # 加载miRNA通过word2vec训练训练好的词向量

            AllCircRNASequence = []
            ReadMyCsv1(AllCircRNASequence, f'Dataset/{args.dataset}/circRNA_order.csv')  # 加载所有circRNA
            CircEmbedding = []
            ReadMyCsv3(CircEmbedding, f'Dataset/{args.dataset}/circRNAEmbedding.csv')  # 加载circRNA通过word2vec训练训练好的词向量


            #     CircEmbeddingFeature = np.array(
            #         GenerateEmbeddingFeature(AllCircRNASequence, CircEmbedding, 64)).reshape(-1, 64, 64)  # 2115,1,64,64
            #     miRNAEmbeddingFeature = np.array(GenerateEmbeddingFeature(AllMiSequence, miRNAEmbedding, 20)).reshape(
            #         -1, 20, 64)  # 821,1,64,64
            #     np.savez(f"../../Dataset/{args.dataset}/EmbeddingFeature.npz", circRNA=CircEmbeddingFeature,
            #              miRNA=miRNAEmbeddingFeature)

            data = np.load(f"Dataset/{args.dataset}/EmbeddingFeature.npz")
            CircEmbeddingFeature = data['circRNA']
            miRNAEmbeddingFeature = data['miRNA']

            CircEmbeddingFeature = torch.tensor(CircEmbeddingFeature, dtype=torch.float).to(args.device)
            miRNAEmbeddingFeature = torch.tensor(miRNAEmbeddingFeature, dtype=torch.float).to(args.device)

            model = NGCF(n_users,
                         n_items,
                         norm_adj,
                         CircEmbeddingFeature,
                         miRNAEmbeddingFeature,
                         args).to(args.device)


            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            max_test_AUC = 0
            max_test_Accuracy = 0
            max_test_MCC = 0
            max_test_Sensitivity = 0
            max_test_Precision = 0
            max_test_Specificity = 0
            max_test_auprs = 0
            max_epoch = 0

            os.makedirs('result/{}'.format(args.dataset), exist_ok=True)

            with open(f"result/{args.dataset}/result-all-{CounterT}.txt", 'a', encoding='utf-8') as file:
                for epoch in range(args.epoch):

                    print(f"epoch: {epoch}")
                    loss, specificity, precision, sensitivity, mcc, accuracy, roc_auc, aupr = model_train(model, optimizer,
                                                                                                    criterion,
                                                                                                    train_loader)
                    print(
                        f"Train_Loss: {loss}  Train_Specificity: {specificity}  Train_Precision: {precision}  Train_Sensitivity: {sensitivity}  Train_MCC: {mcc}  Train_Accuracy: {accuracy}  Train ROC AUC: {roc_auc}  Train AUPR:{aupr}")

                    loss, specificity, precision, sensitivity, mcc, accuracy, roc_auc, aupr  = model_test(model, criterion,
                                                                                                   train_loader)
                    print(
                        f"Test_Loss: {loss}  Test_Specificity: {specificity}  Test_Precision: {precision}  Test_Sensitivity: {sensitivity}  Test_MCC: {mcc}  Test_Accuracy: {accuracy}  Test ROC AUC: {roc_auc}  Train AUPR:{aupr}")

                    file.write(f"epoch: {epoch}\n")
                    file.write(
                        f"Train_Loss: {loss}  Train_Specificity: {specificity}  Train_Precision: {precision}  Train_Sensitivity: {sensitivity}  Train_MCC: {mcc}  Train_Accuracy: {accuracy}  Train ROC AUC: {roc_auc}  Train AUPR:{aupr}\n")
                    file.write(
                        f"Test_Loss: {loss}  Test_Specificity: {specificity}  Test_Precision: {precision}  Test_Sensitivity: {sensitivity}  Test_MCC: {mcc}  Test_Accuracy: {accuracy}  Test ROC AUC: {roc_auc}  Train AUPR:{aupr}\n")

                    if max_test_AUC <= roc_auc:
                        max_epoch = epoch
                        max_test_AUC = roc_auc
                        max_test_Accuracy = accuracy
                        max_test_MCC = mcc
                        max_test_Sensitivity = sensitivity
                        max_test_Precision = precision
                        max_test_Specificity = specificity
                        max_test_aupr = aupr
                    #
                    #     torch.save(model.state_dict(), f"result/{args.dataset}/Bestmodel.pkl")
                    #     print('save the weights in path: ', f"result/{args.dataset}/Bestmodel.pkl")
                    print('\n')

                with open(f"result/{args.dataset}/result.txt", 'a', encoding='utf-8') as f:
                    f.write("dataset:{}\n".format(args.dataset))
                    f.write(
                        f'epoch:{max_epoch}  Test_Specificity: {max_test_Specificity:2.4f}  Test_Precision: {max_test_Precision:2.4f}  Test_Sensitivity: {max_test_Sensitivity:2.4f}  Test_MCC: {max_test_MCC:2.4f}  Test_Accuracy: {max_test_Accuracy:2.4f}  Test ROC AUC: {max_test_AUC:2.4f}  Test AUPR: {max_test_aupr}\n')
