import torch
from torch.utils.data import DataLoader

import argparse
from dgl import model_zoo
import rdkit
import numpy as np

from jtnn.jtnn import chemutils, datautils, mol_tree, nnutils
import json
import os


def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)



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
    model.eval()
    print("Model #Params: %dK" %
          (sum([x.nelement() for x in model.parameters()]) / 1000,))


    return model, args


def reconstruct():
    model, args = setup()
    dataset = datautils.JTNNDataset(data=args['train'], vocab=args['vocab'], training=False)

    dataset.training = False
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=datautils.JTNNCollator(dataset.vocab, False),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        )

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    # this only corresponds to the Section 3.1 of the paper.

    acc = 0.0
    tot = 0
    with torch.no_grad():
        for it, batch in enumerate(dataloader):
            tot += 1

            gt_smiles = batch['mol_trees'][0].smiles

            model.move_to_cuda(batch)
            try:
                _, tree_vec, mol_vec = model.encode(batch)

                tree_mean = model.T_mean(tree_vec)
                # Following Mueller et al.
                tree_log_var = -torch.abs(model.T_var(tree_vec))
                mol_mean = model.G_mean(mol_vec)
                # Following Mueller et al.
                mol_log_var = -torch.abs(model.G_var(mol_vec))

                epsilon = torch.randn(1, model.latent_size // 2) #.cuda()
                tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
                epsilon = torch.randn(1, model.latent_size // 2) #.cuda()
                mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
                dec_smiles = model.decode(tree_vec, mol_vec)

                if dec_smiles == gt_smiles:
                    acc += 1
            except Exception as e:
                print("Failed to encode: {}".format(gt_smiles))
                print(e)

            if it % 20 == 1:
                print("Progress {}/{}; Current Reconstruction Accuracy: {:.4f}".format(it,
                                                                                       len(dataloader), acc / tot))
    return acc / tot


def latent_exp(output_path=None):
    """
    Directly sampling from the latent space instead of doing a reconstruction of the original input
    :return: smiles molecules
    """
    # 450 is the correct latent dim of the pretrained model
    # the following is 1 iteration of sampling and decoding

    epoch = 50
    model, args =setup()
    hidden_size = args['hidden_size']
    for i in range(epoch):
        tree_vec = nnutils.create_var(torch.randn(1, hidden_size), False)
        mol_vec = nnutils.create_var(torch.randn(1, hidden_size), False)
        tree_vec, mol_vec, z_mean, z_log_var = model.sample(tree_vec,  mol_vec)
        try:
            smiles = model.decode(tree_vec, mol_vec)

            if output_path is not None:
                # Write to log file
                with open(os.path.join(output_path, 'latent_gen.txt'), 'a+') as fp:
                    line = '{},{}\n'.format(i, smiles)
                    fp.write(line)
        except Exception as e:
            print("Failed to encode in latent space sampling")
            print(e)

if __name__ == '__main__':
    latent_sample = latent_exp()
