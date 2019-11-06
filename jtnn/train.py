import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dgl import model_zoo
from torch.utils.data import DataLoader

import sys
import argparse
import rdkit
import numpy as np
from jtnn.jtnn import chemutils, datautils, mol_tree, nnutils
import json
import os

torch.multiprocessing.set_sharing_strategy('file_system')


def worker_init_fn(id_):
    # this was a function included in the dgl repo
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)



worker_init_fn(None)

def setup():
    with open('expts/config_single_exp.json') as fp:
        ap = json.load(fp)

    worker_init_fn(None)

    args = ap

    dataset = datautils.JTNNDataset(data=args['train'], vocab=args['vocab'], training=False)
    vocab_file = dataset.vocab_file

    hidden_size = int(args['hidden_size'])
    latent_size = int(args['latent_size'])
    depth = int(args['depth'])

    model = model_zoo.chem.DGLJTNNVAE(vocab_file=vocab_file,
                                      hidden_size=hidden_size,
                                      latent_size=latent_size,
                                      depth=depth)

    if args['model_path'] is not None:
        model.load_state_dict(torch.load(args['model_path']))
    else:
        model = model_zoo.chem.load_pretrained("JTNN_ZINC")

    model = nnutils.cuda(model)
    print("Model #Params: %dK" %
          (sum([x.nelement() for x in model.parameters()]) / 1000,))


    return model, args


def train(output_path=None):
    MAX_EPOCH = 2 #100
    PRINT_ITER = 20

    model, args =  setup()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()
    dataset = datautils.JTNNDataset(data=args['train'], vocab=args['vocab'], training=True)
    if args['model_path'] is not None:
        model.load_state_dict(torch.load(args['model_path']))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    subset_indices = np.arange(50)
    dataset.training = True
    dataloader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=datautils.JTNNCollator(dataset.vocab, True),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        sampler=torch.utils.data.SubsetRandomSampler(subset_indices))

    for epoch in range(MAX_EPOCH):
        word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0

        for it, batch in enumerate(dataloader):
            model.zero_grad()
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, args['beta'])
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Loss: %.6f" % (
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc, loss.item()))
                word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0
                sys.stdout.flush()

            if (it + 1) % 1500 == 0:  # Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                torch.save(model.state_dict(),
                           args.save_path + "/model.iter-%d-%d" % (epoch, it + 1))

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        if output_path is not None:
            torch.save(model.state_dict(), os.path.join(output_path, "model.iter-" + str(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(args['save_path'], "model.iter-" + str(epoch)))

if __name__ == '__main__':
    train()


