# https://github.com/cnielly/prototypical-networks-omniglot/blob/master/prototypical_networks_pytorch_omniglot.ipynb
# https://github.com/jakesnell/prototypical-networks

import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from CERT.utils import utils
import os
from copy import copy

from sklearn.metrics import classification_report, f1_score, roc_auc_score, average_precision_score


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class classification_lstm(nn.Module):
    def __init__(self, options):
        super(classification_lstm, self).__init__()
        self.options = options
        self.input_dim = options['input_dim']
        self.emb_dim = options['emb_dim']
        self.out_dim = options['out_dim']
        self.h0 = options['hid_dim0']
        self.h1 = options['hid_dim1']
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim, padding_idx=0)
        self.embeddingbag = nn.EmbeddingBag(self.input_dim, self.emb_dim, padding_idx=0, mode='mean')
        self.rnn = nn.LSTM(self.emb_dim, self.out_dim, 1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.emb_dim, self.out_dim)
        # self.fc = nn.Sequential(
        #         nn.Linear(self.emb_dim, self.h0),
        #         nn.ReLU(),
        #         nn.Dropout(0.3),
        #         nn.Linear(self.h0, self.h1),
        #         nn.ReLU(),
        #         nn.Dropout(0.3),
        #         nn.Linear(self.h1, self.out_dim)
        #     ).to(options['device'])

    def forward(self, input, lens):
        embedded = self.dropout(self.embedding(input))
        output, _ = self.rnn(embedded)
        output = torch.sum(output, dim=1)
        output = output/lens
        return output

    # def forward(self, input, lens):
    #     emb = self.embeddingbag(input)
    #     return self.fc(emb)

def load_protonet_lstm(options):
    """
    Loads the prototypical network model
    Arg:
        x_dim (tuple): dimension of input windows
        hid_dim (int): dimension of hidden layers
    Returns:
        Model (Class ProtoNet)
    """

    encoder = classification_lstm(options)

    return ProtoNet(encoder, options)


def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): dataset of seqs
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
            (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
            (int): n_way
            (int): n_support
            (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        if len(datax_cls) < (n_support + n_query):
            temp = datax_cls.copy()
            for _ in range(int(n_query+n_support)//(len(datax_cls))+1):
                datax_cls = np.append(datax_cls, datax_cls, axis=0)
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        if isinstance(sample_cls[0][0], list):
            sample.append([i[0] for i in sample_cls])
        else:
            sample.append([i for i in sample_cls])
    sample = np.array(sample)
    return ({
        'seqs': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


def get_iter(x_and_len, y, batch_size=1024, shuffle=False):
    X, lens = x_and_len
    if y is None:
        y = [-1 for _ in range(len(X))]
    dataset = LogDataset(X, lens, y)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size, worker_init_fn=np.random.seed(42))
    return iter


class LogDataset(Dataset):
    def __init__(self, seqs, lens, ys):
        super().__init__()
        self.seqs = seqs
        self.lens = lens
        self.ys = ys

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.seqs[idx], self.lens[idx], self.ys[idx]


def list_to_tensor(lst):
    if isinstance(lst[0][0], list):
        lst = [i[0] for i in lst]
    x_lens = [len(i) for i in lst]
    max_len = max(x_lens)
    for i in range(len(lst)):
        if len(lst[i]) < max_len:
            lst[i].extend([0 for _ in range(max_len - len(lst[i]))])
    lst = torch.stack([torch.tensor(i) for i in lst])
    return lst, torch.tensor(x_lens).reshape(-1, 1)


class ProtoNet(nn.Module):
    def __init__(self, encoder, options):
        """
        Args:
            encoder : LSTM encoding the sequence in samples
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.options = options
        self.device = options['device']
        self.encoder = encoder.to(self.device)
        self.input_dim = options['input_dim']
        self.emb_dim = options['emb_dim']

    def load_encoder(self, path=None):
        if path == None:
            print('Please provide a path')
        else:
            # print(f'Loading model from: {path}')
            self.encoder = torch.load('./CERT/saved_models/' + path)
            self.encoder.to(self.device)

    def set_forward_loss(self, sample, retrain=0):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_seqs = sample['seqs']
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support = sample_seqs[:, :n_support]
        x_query = sample_seqs[:, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.to(self.device)

        if isinstance(x_query[0][0], list):
            x = np.concatenate((x_support.reshape(-1), x_query.reshape(-1)), axis=0)
        else:
            x = np.concatenate((x_support.reshape(n_way*n_support, -1), x_query.reshape(n_way*n_support, -1)), axis=0)

        x_lens = [len(i) for i in x]
        max_len = max(x_lens)
        for i in range(len(x)):
            if len(x[i]) < max_len:
                x[i].extend([0 for _ in range(max_len-len(x[i]))])

        x = torch.reshape(torch.tensor(np.concatenate(x)), (len(x_lens), max_len)).to(self.device)
        x_lens = torch.tensor(x_lens).reshape(-1, 1).to(self.device)

        if retrain:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder.forward(x, x_lens)
                z_dim = z.size(-1)
                z_proto_before = z[:n_way * n_support].view(n_way, n_support, z_dim).mean(1)
            self.encoder.train()

        z = self.encoder.forward(x, x_lens)
        z_dim = z.size(-1)
        z_proto = z[:n_way * n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way * n_support:]

        # compute distances
        dists = euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        if retrain:
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() + 0.01*torch.sum(euclidean_dist(z_proto, z_proto_before))
        else:
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        target_inds.cpu()
        x.cpu()
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat,
            'protoes': z_proto
        }

    def train(self, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size, path='PN.pth', retrain=0):
        """
        Trains the protonet
        Args:
            model
            optimizer
            train_x (np.array): images of training set
            train_y(np.array): labels of training set
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
            max_epoch (int): max epochs to train on
            epoch_size (int): episodes per epoch
        """
        # divide the learning rate by 2 at each epoch, as suggested in paper
        self.encoder.train()
        self.encoder.to(self.device)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
        epoch = 0  # epochs done so far
        stop = False  # status to know when to stop

        while epoch < max_epoch and not stop:
            running_loss = 0.0
            running_acc = 0.0
            min_loss = 1000

            for episode in range(epoch_size):
                sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
                optimizer.zero_grad()
                loss, output = self.set_forward_loss(sample, retrain)
                running_loss += output['loss']
                running_acc += output['acc']
                loss.backward()
                optimizer.step()

            epoch_loss = running_loss / epoch_size
            epoch_acc = running_acc / epoch_size
            if min_loss >= epoch_loss:
                min_loss = copy(epoch_loss)
                torch.save(self.encoder, './CERT/saved_models/' + path)
            # print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
            epoch += 1
            scheduler.step()

    def test(self, train_iter, test_iter, path='PN.pth', validation=0):
        self.encoder = torch.load('./CERT/saved_models/' + path)
        self.encoder.to(self.device)
        self.encoder.eval()
        embs = []
        y = []
        # print('Compute the centers with training dataset')
        for batch in train_iter:
            x = batch[0].to(self.device)
            x_lens = batch[1].to(self.device)
            labels = batch[2]
            z = self.encoder.forward(x, x_lens)
            embs.extend(z.detach().cpu().numpy())
            y.extend(list(labels.numpy()))
        df_temp = pd.DataFrame(embs)
        df_temp['y'] = y
        protoes = torch.tensor(df_temp.groupby('y').mean().values).to(self.device)
        del df_temp

        total_loss = 0
        y_pred = []
        y_true = []
        y_vals = []
        y_all = []
        y_dist = []
        y_normal = []
        # print('Make prediction with centers')
        for batch in test_iter:
            x = batch[0].to(self.device)
            x_lens = batch[1].to(self.device)
            y_true.extend(list(batch[2]))
            z = self.encoder.forward(x, x_lens)
            dists = euclidean_dist(z, protoes)
            y_dist.extend(list(torch.min(dists, dim=1)[0].detach().cpu().numpy()))
            if validation == 0:
                y_val, _ = F.softmax(-dists, dim=1).max(1)
                y_val = y_val.detach().cpu().numpy()
            else:
                y_val = F.softmax(-dists, dim=1).detach().cpu().numpy()[list(range(len(batch[2]))), list(batch[2])].reshape(-1)
                assert len(y_val) == len(batch[1]), 'Length for y_val is different than org'
            log_p_y = F.log_softmax(-dists, dim=1)
            _, y_hat = log_p_y.max(1)
            y_all.extend(list(F.softmax(-dists, dim=1).detach().cpu().numpy()))
            y_pred.extend(list(y_hat.detach().cpu().numpy()))
            y_vals.extend(list(y_val))
            y_normal.extend(list(dists.detach().cpu().numpy()[:, 0].reshape(-1)))
        # print(multilabel_confusion_matrix(y_true, y_pred))
        # print(classification_report(y_true, y_pred, digits=5))

        return {'y_pred': y_pred,
                'y_true': y_true,
                'y_vals': y_vals,
                'y_all': y_all,
                'y_dist': y_dist,
                'y_normal': y_normal,
                'protoes': protoes}, f1_score(y_true, y_pred, average='macro')

    def embedding(self, train_iter, test_iter, path='PN.pth', validation=0):
        self.encoder = torch.load('./CERT/saved_models/' + path)
        self.encoder.to(self.device)
        self.encoder.eval()
        embs = []
        y = []
        # print('Compute the centers with training dataset')
        for batch in train_iter:
            x = batch[0].to(self.device)
            x_lens = batch[1].to(self.device)
            labels = batch[2]
            z = self.encoder.forward(x, x_lens)
            embs.extend(z.detach().cpu().numpy())
            y.extend(list(labels.numpy()))
        df_temp = pd.DataFrame(embs)
        df_temp['y'] = y
        protoes = torch.tensor(df_temp.groupby('y').mean().values).to(self.device)
        del df_temp

        total_loss = 0
        x_vals = []
        y_pred = []
        y_true = []
        y_vals = []
        y_all = []
        y_dist = []
        y_normal = []
        # print('Make prediction with centers')
        for batch in test_iter:
            x = batch[0].to(self.device)
            x_lens = batch[1].to(self.device)
            y_true.extend(list(batch[2]))
            z = self.encoder.forward(x, x_lens)
            x_vals.extend(z.cpu().detach().numpy())
            dists = euclidean_dist(z, protoes)
            y_dist.extend(list(torch.min(dists, dim=1)[0].detach().cpu().numpy()))
            if validation == 0:
                y_val, _ = F.softmax(-dists, dim=1).max(1)
                y_val = y_val.detach().cpu().numpy()
            else:
                y_val = F.softmax(-dists, dim=1).detach().cpu().numpy()[
                    list(range(len(batch[2]))), list(batch[2])].reshape(-1)
                assert len(y_val) == len(batch[2]), 'Length for y_val is different than org'
            log_p_y = F.log_softmax(-dists, dim=1)
            _, y_hat = log_p_y.max(1)
            y_all.extend(list(F.softmax(-dists, dim=1).detach().cpu().numpy()))
            y_pred.extend(list(y_hat.detach().cpu().numpy()))
            y_vals.extend(list(y_val))
            y_normal.extend(list(dists.detach().cpu().numpy()[:,0].reshape(-1)))
        # print(multilabel_confusion_matrix(y_true, y_pred))
        # print(classification_report(y_true, y_pred, digits=5))

        return {'x_vals': x_vals,
                'y_pred': y_pred,
                'y_true': y_true,
                'y_vals': y_vals,
                'y_all': y_all,
                'y_dist': y_dist,
                'y_normal': y_normal,
                'protoes': protoes}


class ProtoTrainer(object):
    def __init__(self, options):
        super(ProtoTrainer, self).__init__()
        self.input_dim = options['input_dim']
        self.options = options
        self.out_dim = options['out_dim']
        self.best_net = load_protonet_lstm(options).to(options['device'])
        self.optim_best = optim.Adam(self.best_net.parameters(), lr=options['lr'])
        self.n_way = options['n_ways']
        self.n_support = options['n_support']
        self.n_query = options['n_query']
        self.n_valid = options['n_valid']
        self.hard_boundary = 0.5
        self.r_hard = options['r_hard']
        self.max_epoch = options['max_epoch']
        self.epoch_size = options['epoch_size']
        self.train_iter = None
        self.num_samples = options['num_samples']
        self.seed = options['random_seed']
        self.current_net = load_protonet_lstm(options).to(options['device'])
        self.optim_current = optim.Adam(self.current_net.parameters(), lr=options['lr'])
        self.temp_net = load_protonet_lstm(options).to(options['device'])
        self.optim_temp = optim.Adam(self.temp_net.parameters(), lr=options['lr'])
        self.validation_size = options['validation_size']
        self.name = '' + str(options['r_ad_alpha']) + str(options['r_cl_alpha']) + str(self.seed) + str(
            self.validation_size) + str(self.n_way)

    def training_first(self, df_seen, df_unseen, df_sup, test_x, test_y):
        df_seen = pd.concat([df_seen, df_sup]).reset_index(drop=True)
        for i in range(len(set(df_seen['y_true']))):
            if i == 0:
                df_seen_eval = df_seen.loc[df_seen['y_true'] == i].sample(self.n_valid, replace=False)
            else:
                df_seen_eval = pd.concat([df_seen_eval, df_seen.loc[df_seen['y_true'] == i].sample(self.n_valid, replace=False)], axis=0)
        df_seen.drop(list(set(df_seen_eval.index.values)), inplace=True)
        df_seen.reset_index(drop=True, inplace=True)
        df_seen_eval.reset_index(drop=True, inplace=True)

        train_x = df_seen.iloc[:, :-3].values
        train_y = df_seen['y_true'].values
        self.best_net.train(self.optim_best, train_x, train_y, self.n_way, self.n_support, self.n_query,
                                  self.max_epoch, self.epoch_size, path='best'+self.name+'.pth')
        train_iter = get_iter(list_to_tensor(train_x), train_y)
        self.train_iter = train_iter
        seen_test_iter = get_iter(list_to_tensor(df_seen_eval.iloc[:, :-3].values), df_seen_eval['y_true'].values)
        seen_result, seen_f1 = self.best_net.test(train_iter, seen_test_iter, path='best'+self.name+'.pth', validation=1)
        test_iter = get_iter(list_to_tensor(test_x), test_y)
        test_result, test_f1 = self.best_net.test(train_iter, test_iter, path='best'+self.name+'.pth')

        # print result for the first iterator
        print('First time result with only few shot samples')
        print(classification_report(test_result['y_true'], test_result['y_pred'], digits=5))
        print('Classification AUC-ROC: {:.5f}'.format(
            roc_auc_score(test_result['y_true'], test_result['y_all'], multi_class='ovr')))
        test_true_ab = np.array(test_result['y_true']).copy()
        test_true_ab[test_true_ab >= 1] = 1
        test_pred_ab = np.array(test_result['y_pred']).copy()
        test_pred_ab[test_pred_ab >= 1] = 1
        print('Anomaly Detection F1:')
        print(classification_report(test_true_ab, test_pred_ab, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_true_ab, test_pred_ab)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_true_ab, test_pred_ab)))
        print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(
            utils.getfpr95tpr(y_true=test_true_ab, dist=test_result['y_normal'])))

        unseen_iter = get_iter(list_to_tensor(df_unseen.iloc[:, :-3].values), df_unseen['y_true'].values)
        dic = self.best_net.embedding(train_iter, unseen_iter, path='best'+self.name+'.pth')
        df_unseen['dist'] = dic['y_dist']
        df_unseen['y_pred'] = dic['y_pred']
        seen_true_ad = np.array(seen_result['y_true']).copy()
        seen_true_ad[seen_true_ad >= 1] = 1
        seen_pred_ad = np.array(seen_result['y_pred']).copy()
        seen_pred_ad[seen_pred_ad >= 1] = 1
        seen_f1_ad = f1_score(y_true=seen_true_ad, y_pred=seen_pred_ad)
        hard_sample = [ind for ind, val in enumerate(seen_result['y_vals']) if val < self.hard_boundary]
        if len(hard_sample) == 0:
            seen_prob = 1
        else:
            seen_prob = np.mean([i for i in seen_result['y_vals'] if i < self.hard_boundary])

        return df_seen, df_seen_eval, df_unseen, seen_f1_ad, seen_f1, hard_sample, seen_prob

    def training_baseline(self, df_seen, df_seen_eval, df_unseen, test_x, test_y):
        n_clusters = set(df_unseen['y_pred'].values)
        for ind, val in enumerate(n_clusters):
            if ind == 0:
                n = int(0.1 * len(df_unseen.loc[df_unseen['y_pred'] == val]))
                df_samples = df_unseen.loc[df_unseen['y_pred'] == val].nsmallest(n, 'dist')
            else:
                n = int(0.1 * len(df_unseen.loc[df_unseen['y_pred'] == val]))
                df_samples = pd.concat([df_samples, df_unseen.loc[df_unseen['y_pred'] == val].nsmallest(n, 'dist')])

        df_seen_0 = pd.concat([df_seen, df_seen_eval, df_samples], axis=0).reset_index(drop=True)
        # df_seen_0 = pd.concat([df_seen, df_seen_eval, df_unseen], axis=0).reset_index(drop=True)
        train_x = df_seen_0.iloc[:, :-3].values
        train_y = df_seen_0['y_pred'].values
        self.temp_net.train(self.optim_temp, train_x, train_y, self.n_way, self.n_support, self.n_query,
                                  self.max_epoch, self.epoch_size, path='baseline'+self.name+'.pth')

        train_iter = get_iter(list_to_tensor(train_x), train_y)
        test_iter = get_iter(list_to_tensor(test_x), test_y)
        test_result, test_f1 = self.temp_net.test(train_iter, test_iter, path='baseline'+self.name+'.pth')

        # print result for the first iterator
        print('Pseudo-label results as baseline')
        print(classification_report(test_result['y_true'], test_result['y_pred'], digits=5))
        print('Classification AUC-ROC: {:.5f}'.format(
            roc_auc_score(test_result['y_true'], test_result['y_all'], multi_class='ovr')))
        test_true_ab = np.array(test_result['y_true']).copy()
        test_true_ab[test_true_ab >= 1] = 1
        test_pred_ab = np.array(test_result['y_pred']).copy()
        test_pred_ab[test_pred_ab >= 1] = 1
        print('Anomaly Detection F1:')
        print(classification_report(test_true_ab, test_pred_ab, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_true_ab, test_pred_ab)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_true_ab, test_pred_ab)))
        print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(
            utils.getfpr95tpr(y_true=test_true_ab, dist=test_result['y_normal'])))

        self.temp_net.to('cpu')

    def training_before(self, df_seen, df_seen_eval, df_unseen, df_selected, max_reward, i_iterator=0):
        if max_reward == (self.options['r_ad_alpha']+self.options['r_cl_alpha']+self.r_hard):
            df_seen = pd.concat([df_seen, df_seen_eval], axis=0)
            df_seen.reset_index(drop=True, inplace=True)
            for i in range(len(set(df_seen['y_true']))):
                if i == 0:
                    if self.n_valid == self.validation_size:
                        df_seen_eval = df_seen.loc[df_seen['y_true'] == i].copy()
                    else:
                        df_seen_eval = df_seen.loc[df_seen['y_true'] == i].sample(self.n_valid, replace=False, random_state=i_iterator)
                else:
                    if self.n_valid == self.validation_size:
                        df_seen_eval = pd.concat([df_seen_eval, df_seen.loc[df_seen['y_true'] == i]], axis=0)
                    else:
                        df_seen_eval = pd.concat([df_seen_eval, df_seen.loc[df_seen['y_true'] == i].sample(self.n_valid, replace=False, random_state=i_iterator)], axis=0)
            df_seen.drop(list(set(df_seen_eval.index.values)), inplace=True)
            df_seen.reset_index(drop=True, inplace=True)
            df_seen_eval.reset_index(drop=True, inplace=True)
        if len(df_selected) == 0:
            train_x = df_seen.iloc[:, :-3].values
            train_y = df_seen['y_true'].values
        else:
            train_x = np.concatenate((df_seen.iloc[:, :-3].values, df_selected.iloc[:, :-3].values), axis=0)
            train_y = np.concatenate((df_seen['y_true'].values, df_selected['y_pred'].values), axis=None)

        train_iter = get_iter(list_to_tensor(train_x), train_y)
        self.train_iter = train_iter
        seen_test_iter = get_iter(list_to_tensor(df_seen_eval.iloc[:, :-3].values), df_seen_eval['y_true'].values)
        seen_result, seen_f1 = self.best_net.test(train_iter, seen_test_iter, path='best'+self.name+'.pth', validation=1)
        unseen_iter = get_iter(list_to_tensor(df_unseen.iloc[:, :-3].values), df_unseen['y_true'].values)
        dic = self.best_net.embedding(train_iter, unseen_iter, path='best'+self.name+'.pth')
        df_unseen['dist'] = dic['y_dist']
        df_unseen['y_pred'] = dic['y_pred']

        seen_true_ad = np.array(seen_result['y_true']).copy()
        seen_true_ad[seen_true_ad >= 1] = 1
        seen_pred_ad = np.array(seen_result['y_pred']).copy()
        seen_pred_ad[seen_pred_ad >= 1] = 1
        seen_f1_ad = f1_score(y_true=seen_true_ad, y_pred=seen_pred_ad)
        hard_sample = [ind for ind, val in enumerate(seen_result['y_vals']) if val < self.hard_boundary]
        if len(hard_sample) == 0:
            seen_prob = 1
        else:
            seen_prob = np.mean([i for i in seen_result['y_vals'] if i < self.hard_boundary])
        return df_seen, df_seen_eval, df_unseen, df_selected, seen_f1_ad, seen_f1, hard_sample, seen_prob

    def in_embedding(self, df_samples, iter):
        samples_iter = get_iter(list_to_tensor(df_samples.iloc[:, :-3].values), df_samples['y_pred'].values,
                                shuffle=False)
        result = self.best_net.embedding(self.train_iter, samples_iter, 'best'+self.name+'.pth')
        df = pd.DataFrame(result['x_vals'])
        df['prob'] = result['y_vals']
        return df

    def training_after(self, df_seen, df_selected, df_selected_episode, df_seen_eval, hard_sample):
        if len(df_selected) == 0:
            train_x = np.concatenate((df_seen.iloc[:, :-3].values, df_selected_episode.iloc[:, :-3].values), axis=0)
            train_y = np.concatenate((df_seen['y_true'], df_selected_episode['y_pred'].values), axis=None)
        else:
            train_x = np.concatenate((df_seen.iloc[:, :-3].values, df_selected.iloc[:, :-3].values, df_selected_episode.iloc[:, :-3].values), axis=0)
            train_y = np.concatenate((df_seen['y_true'], df_selected['y_pred'].values, df_selected_episode['y_pred'].values), axis=None)
        self.current_net.load_encoder('best'+self.name+'.pth')
        self.optim_current = optim.Adam(self.current_net.parameters(), lr=self.options['lr'])
        self.current_net.train(self.optim_current, train_x, train_y, self.n_way, self.n_support, self.n_query,
                                  int(self.max_epoch/2), self.epoch_size, path='current'+self.name+'.pth', retrain=1)
        train_iter = get_iter(list_to_tensor(train_x), train_y)
        seen_test_iter = get_iter(list_to_tensor(df_seen_eval.iloc[:, :-3].values), df_seen_eval['y_true'].values)
        seen_result, seen_f1 = self.current_net.test(train_iter, seen_test_iter, path='current'+self.name+'.pth', validation=1)

        seen_true_ad = np.array(seen_result['y_true']).copy()
        seen_true_ad[seen_true_ad >= 1] = 1
        seen_pred_ad = np.array(seen_result['y_pred']).copy()
        seen_pred_ad[seen_pred_ad >= 1] = 1
        seen_f1_ad = f1_score(y_true=seen_true_ad, y_pred=seen_pred_ad)
        if len(hard_sample) == 0:
            seen_prob = 1
        else:
            seen_prob = np.mean(np.array(seen_result['y_vals'])[hard_sample])
        return seen_f1_ad, seen_f1, seen_prob


    def save_best_model(self):
        os.rename('./CERT/saved_models/current'+str(self.name)+'.pth', './CERT/saved_models/temp_best'+str(self.name)+'.pth')

    def remove_model(self):
        os.remove('./CERT/saved_models/temp_best' + self.name + '.pth')

    def update_best_model(self):
        os.remove('./CERT/saved_models/best'+str(self.name)+'.pth')
        os.rename('./CERT/saved_models/temp_best'+str(self.name)+'.pth', './CERT/saved_models/best'+str(self.name)+'.pth')

    def final_training_testing(self, df_seen, df_selected, test_x, test_y, final=0):
        if len(df_selected) == 0:
            train_x = df_seen.iloc[:, :-3].values
            train_y = df_seen['y_true'].values
        else:
            train_x = np.concatenate((df_seen.iloc[:, :-3].values, df_selected.iloc[:, :-3].values), axis=0)
            train_y = np.concatenate((df_seen['y_true'].values, df_selected['y_pred']), axis=None)
        train_iter = get_iter(list_to_tensor(train_x), train_y)
        test_iter = get_iter(list_to_tensor(test_x), test_y)
        test_result, test_f1 = self.best_net.test(train_iter, test_iter, path='best'+self.name+'.pth')

        # print result for the first iterator
        print(classification_report(test_result['y_true'], test_result['y_pred'], digits=5))
        print('Classification AUC-ROC: {:.5f}'.format(
            roc_auc_score(test_result['y_true'], test_result['y_all'], multi_class='ovr')))
        test_true_ab = np.array(test_result['y_true']).copy()
        test_true_ab[test_true_ab >= 1] = 1
        test_pred_ab = np.array(test_result['y_pred']).copy()
        test_pred_ab[test_pred_ab >= 1] = 1
        print('Anomaly Detection report:')
        print(classification_report(test_true_ab, test_pred_ab, digits=5))
        print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(test_true_ab, test_pred_ab)))
        print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(test_true_ab, test_pred_ab)))
        if final:
            print('Anomaly Detection FPR-AT-95-TPR: {:.5f}'.format(utils.getfpr95tpr(y_true=test_true_ab, dist=test_result['y_normal'])))
        if len(df_selected) > 0:
            print(df_selected.groupby(['y_pred', 'y_true']).count())



