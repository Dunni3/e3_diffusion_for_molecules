# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
import argparse
from configs.datasets_config import qm9_with_h, qm9_without_h
from qm9 import dataset
from qm9.models import get_model

from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from os.path import join
from qm9.sampling import sample_chain, sample
from configs.datasets_config import get_dataset_info

from types import SimpleNamespace
import time
import numpy as np
import pickle

def sample_batch(n_atoms: int, n_molecules: int, args, device, model, dataset_info):

    nodesxsample = torch.ones([n_molecules], dtype=torch.int64)*n_atoms

    # sample molecules
    one_hot, charges, x, node_mask = sample(args, device, model, dataset_info, nodesxsample=nodesxsample)

    return one_hot, charges, x, node_mask

def compute_avg_com_dist(molecules: dict) -> float:
    """Computes the average distance of atoms from the center of mass of each molecule, for a batch of molecules.
       Note that this function assumes that all molecules are the same size. Node masks are not taken into account."""
    positions = molecules['x'] # has shape (n_molecules, n_atoms, 3)

    # compute center of mass for each molecules
    com = torch.mean(positions, dim=1, keepdim=True) # shape (n_molecules, 1, 3)

    # compute distance from COM for every atom
    com_displacement = positions - com # (n_molecules, n_atoms, 3)
    com_dist = torch.norm(com_displacement, dim=2) # (n_molecules, n_atoms)

    # compute average distance from COM
    avg_com_dist = torch.mean(com_dist)
    return float(avg_com_dist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='e3diff size analysis')
    parser.add_argument('--model_path', type=str, default="outputs/edm_qm9",
                    help='Specify model path')
    parser.add_argument('--output', type=str, default='size_analysis_results.pkl',
                        help='filepath of output file')
    parser.add_argument('--min_size', type=int, default=10,
                        help='number of atoms in the smallest molecule sampled')
    parser.add_argument('--max_size', type=int, default=100,
                        help='number of atoms in the largest molecule sampled')
    parser.add_argument('--n_sizes', type=int, default=50,
                        help='number of molecule sizes to test')
    parser.add_argument('--n_reps', type=int, default=50,
                        help='number of molecules to sample for each size')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch_size to use when sampling molecules')

    # manually create evaluation args
    # eval_args = SimpleNamespace(model_path='outputs/edm_qm9', n_tries=10, n_nodes=19)
    eval_args, unparsed_args = parser.parse_known_args()

    # get model args
    model_args_file = join(eval_args.model_path, 'args.pickle')
    with open(model_args_file, 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    # set device??
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    # create folders?
    utils.create_folders(args)
    print(args)

    # get dataset info
    dataset_info = get_dataset_info(args.dataset, args.remove_h)


    # get dataloaders, this is needed for get_model(). 
    # we shouldn't need to load the training dataset into memory in order to sample from the 
    # trained model
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    model, nodes_dist, prop_dist = get_model(
        args, device, dataset_info, dataloaders['train'])
    model.to(device)

    # get diffusion model and load its state dict i guess?
    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn),
                                    map_location=device)

    model.load_state_dict(flow_state_dict)


    # define parameters from command line args
    output_file = eval_args.output
    min_molecule_size: int = eval_args.min_size
    max_molecule_size: int = eval_args.max_size
    n_sizes: int = eval_args.n_sizes
    n_reps: int = eval_args.n_reps
    sample_batch_size: int = eval_args.batch_size

    # array of molecule sizes to be sampled
    molecule_size = np.unique(np.geomspace(
        min_molecule_size, max_molecule_size, num=n_sizes).astype(int))

    output_dict = {
        'molecule_size': molecule_size.copy(),
        'frac_mol_stable': np.zeros(molecule_size.shape, dtype=float),
        'frac_atoms_stable': np.zeros(molecule_size.shape, dtype=float),
        'avg_com_dist': np.zeros(molecule_size.shape, dtype=float)
    }

    # iterate over molecule sizes
    for i, n_atoms in enumerate(molecule_size):

        # create a dictionary that will contain all the batches of
        # sampled molecules for this molecule size
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        
        # sample a batch of molecules with this size
        mols_sampled = 0
        while mols_sampled < n_reps:

            # determine how many molecules we need to sample in this batch
            if n_reps - mols_sampled > sample_batch_size:
                mols_to_sample = sample_batch_size
            else:
                mols_to_sample = n_reps - mols_sampled

            # update the total number of molecules sampled for this molecule size
            mols_sampled += mols_to_sample

            # sample molecules
            one_hot, charges, x, node_mask =  sample_batch(
                n_atoms=n_atoms, 
                n_molecules=mols_to_sample, 
                args=args, 
                device=device, 
                model=model, 
                dataset_info=dataset_info)

            # add this batch to the dict containing other batches
            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach.cpu())

        # concatenate batches of molecules
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

        # compute stability metrics for the sampled molecules
        validity_metrics, rdkit_metrics = analyze_stability_for_molecules(molecules, dataset_info)

        # compute average distance from COM
        avg_com_dist = compute_avg_com_dist(molecules)

        # record metrics for this molecule size
        output_dict['frac_mol_stable'][i] = validity_metrics['mol_stable']
        output_dict['frac_atoms_stable'][i] = validity_metrics['atm_stable']
        output_dict['avg_com_dist'][i] = avg_com_dist

        # write to output file
        with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f)


            