import os, torch, uuid
from torch.utils.data import random_split
from multiprocessing.pool import Pool
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.data.collate import collate
from torch.utils.data import DataLoader
import lightning as L
from scipy.linalg import eigh
from torch_geometric.utils import get_laplacian,to_scipy_sparse_matrix,is_sparse
import numpy as np
import networkx as nx
from .MS_chem import MSMol,MSSpectrum, Formula
from .definition import *
from .utils import NIST_spectrum_iterator, peaks2ndarray
from typing import *
from typing_extensions import Self
from functools import partial

def graph2array(mol_graph:nx.Graph):
    node_h, bond_h = [], []
    edge_index = []
    for (_,atom_prop) in mol_graph.nodes(data=True):
        element, hybri, charge, Hs = atom_prop['element'], atom_prop['hybridization'], atom_prop['charge'], atom_prop['Hs']
        ele_onehot = np.array([element == E for E in ELEMENTS_ONEHOT],dtype=np.float32)
        hybri_onehot = np.array([hybri == H for H in HYBRIDIZATION_ONEHOT],dtype=np.float32)
        node_h.append(np.concatenate([
            ele_onehot,hybri_onehot,
            np.array([charge,Hs],dtype=np.float32),
        ],axis=0))
    for (i,j,bond_prop) in mol_graph.edges(data=True):
        bond_type, in_ring = bond_prop['bond_type'], bond_prop['in_ring']
        bond_onehot = np.array([bond_type==B for B in BOND_ONEHOT])
        bond_onehot = np.concatenate([bond_onehot,np.array([in_ring],dtype=np.float32)],axis=0)
        bond_h.append(bond_onehot)
        bond_h.append(bond_onehot) # di-graph
        edge_index.append((i,j))
        edge_index.append((j,i))
    node_h = np.array(node_h,dtype=np.float32)
    bond_h = np.array(bond_h,dtype=np.float32)
    edge_index = np.transpose(np.array(edge_index,dtype=np.int64))
    return node_h, bond_h, edge_index

def graph_laplacian(edge_index, num_nodes, num_eigen, padding_value=0.):
   
    V = np.empty(shape = (num_nodes, num_eigen), dtype = np.float32)
    D = np.empty(shape = (num_eigen,), dtype = np.float32)
    edge_index = torch.from_numpy(edge_index)
    if num_nodes > 0:
        edge_index, edge_weight = get_laplacian(
            edge_index,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_vals, eig_vecs = eigh(L.toarray())

        idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:,idx])
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,1:num_eigen+1]
        eig_vals = eig_vals[1:num_eigen+1]

        eig_vecs = eig_vecs / np.linalg.norm(eig_vecs,axis=0)
        
        V[:] = padding_value
        V[:,:eig_vecs.shape[1]] = eig_vecs

        D[:] = padding_value
        D[:eig_vals.shape[0]] = eig_vals

    eigvecs = V
    eigvals = D[np.newaxis,:]

    return eigvecs, eigvals

def spectrum_covariates_encoding(spectrum:MSSpectrum) -> np.ndarray:
    instrument_onehot = [spectrum.INSTRUMENT_TYPE == T for T in INSTRUMENT_TYPES]
    precursor_onehot = [spectrum.PRECURSOR_TYPE == T for T in PRECURSOR_TYPES]
    nce = spectrum.get_nce()
    return np.array([*instrument_onehot,*precursor_onehot,nce],dtype=np.float32).reshape([1,6])

class SpectrumData(Data):
    def __init__(
        self,
        x = None,
        edge_index = None,
        edge_attr = None,
        y = None,
        pos = None,
        time = None,
        **kwargs,
    ):
        super().__init__(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,pos=pos,time=time,**kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if is_sparse(value) and ('adj' in key or 'edge_index' in key):
            return (0, 1)
        elif key == 'frag_node_index':
            return 0
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

def spectrum2data(
        spectrum:MSSpectrum,
        loss_formulas:list[Formula],
        include_ring_cuts = True,
        num_eigen:int = 8,
        padding_value = 0.,
    ) -> SpectrumData|None:
    """Transform spectrum into torch_geometric.data.Data.
        graph from the mol linked with spectrum,
        targets from fragment matching result linked with spectrum

    Args:
        spectrum (MSSpectrum): _description_

    Returns:
        Data: _description_
    """
    msmol = MSMol(spectrum.get_mol(),spectrum.precursor)
    node_h, bond_h, edge_index = graph2array(msmol.mol_graph)
    fragments = msmol.get_fragments_from_single_cut(include_ring_cuts=include_ring_cuts)
    frag_node_index = [fragment.ion_nodes for fragment in fragments]
    peak_intensity, fragments_map, mz_matrix,formula_matrix = spectrum.prepare_data_for_training(loss_formulas=loss_formulas)
    raw_peaks = peaks2ndarray(spectrum.raw_peaks) # [num_peaks, 2]
    if peak_intensity is not None: 
        covariates = spectrum_covariates_encoding(spectrum)
        eigvecs, eigvals = graph_laplacian(edge_index=edge_index,num_nodes=len(msmol.mol_graph),num_eigen=num_eigen,padding_value=padding_value)
        graph_data = SpectrumData(
            node_h = torch.from_numpy(node_h),
            edge_index = torch.from_numpy(edge_index),
            edge_attr = torch.from_numpy(bond_h),
            fragments_map = torch.from_numpy(fragments_map), # shape (len(frags),3,len(loss_formulas)
            peak_intensity = torch.from_numpy(peak_intensity), # shape (len(peak)+1,1), first 'peak' empty, real peak from idx 1
            covariates = torch.from_numpy(covariates), # shape [1,6]
            frag_node_index = frag_node_index, # list[list[int]]
            num_frags = len(frag_node_index),
            mz_matrix = torch.from_numpy(mz_matrix), # shape [len(frags),3,len(loss_formulas)]
            eigvecs = torch.from_numpy(eigvecs), # shape = [num_node, num_eigen]
            eigvals = torch.from_numpy(eigvals),  # shape = [1, num_eigen]
            raw_peaks = torch.from_numpy(raw_peaks), 
            formula_matrix = torch.from_numpy(formula_matrix) # shape [len(frags),3,len(loss_formulas),ELE_onehot]
        )
        return graph_data
    else: # some spectrums have no peaks after filtering
        return None

class SpectrumDataBatch(Batch):
    @classmethod
    def from_data_list(
        cls,
        data_list: List,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        data_list = SpectrumDataBatch.add_virtual_and_padding(data_list=data_list)
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)  # type: ignore
        batch._slice_dict = slice_dict  # type: ignore
        batch._inc_dict = inc_dict  # type: ignore

        return batch

    @staticmethod
    def add_virtual_and_padding(data_list: list[SpectrumData]):
        # 1. pad length according to largest frag (with most nodes)
        # +1 to include the added virtual node
        pad_length = max([data.num_nodes for data in data_list]) + 1
        # 2. add virtual node to each data graph
        out_data_list = []
        for data in data_list:
            num_nodes = data.num_nodes
            node_h = torch.zeros(
                size=(num_nodes+1,*data.node_h.shape[1:]),
                dtype=data.node_h.dtype, device=data.node_h.device
            )
            node_h[1:] = data.node_h
            node_mask = torch.ones(
                size = (num_nodes+1,1),
                dtype=data.node_h.dtype, device=data.node_h.device
            )
            node_mask[0] = 0.
            edge_index = data.edge_index + 1 # 
            frag_node_index = torch.zeros(
                size=(len(data.frag_node_index),pad_length),
                dtype=torch.long,device=data.node_h.device
            )
            frag_node_count = torch.zeros(
                size=(len(data.frag_node_index),1),
                dtype=torch.float32,device=data.node_h.device
            )
            for i, node_index in enumerate(data.frag_node_index): 
                target_t = torch.tensor([idx+1 for idx in node_index],dtype=frag_node_index.dtype,device=frag_node_index.device)
                frag_node_index[i,:len(node_index)] = target_t
                frag_node_count[i] = float(len(node_index))
            # 2.1 eigenvecs
            eigvecs = torch.zeros(
                size=(num_nodes+1,data.eigvecs.shape[1]),
                dtype=data.eigvecs.dtype,device=data.eigvecs.device
            )
            eigvecs[1:,:] = data.eigvecs
            
            out_data_list.append(
                SpectrumData(
                    node_h = node_h,
                    node_mask = node_mask,
                    edge_index = edge_index,
                    edge_attr = data.edge_attr,
                    fragments_map = data.fragments_map, # shape (len(frags),3*len(loss_formulas)
                    peak_intensity = data.peak_intensity, # shape (len(peak)+1,1), first 'peak' empty, real peak from idx 1
                    covariates = data.covariates,
                    eigvecs = eigvecs,
                    eigvals = data.eigvals,
                    frag_node_index = frag_node_index,
                    frag_node_count = frag_node_count,
                    num_frags = data.num_frags,
                    mz_matrix = data.mz_matrix,
                    raw_peaks = data.raw_peaks,
                    formula_matrix = data.formula_matrix
            ))
        return out_data_list
      
    def get_batch_fragment(self,device):
        fragment_batch = []
        for (graph_index,num_frag) in zip(range(self.num_graphs),self.num_frags):
            fragment_batch.extend([graph_index]*int(num_frag))
        return torch.tensor(fragment_batch,dtype=torch.int64,device=device)
    
################################## NIST Dataset #############################
# NIST dataset is organized as below:
# NIST_ROOT/
#   Inchi_key1/
#       xxx.pt (Data), xxx.pt (Data) 
#   Inchi_key2/ ...
#       xxx.pt (Data), xxx.pt (Data) 
#############################################################################

class NISTDataset(Dataset):
    def __init__(
            self, 
            nist_dir:str, inchi_keys:list[str] = None, train = True,
            root = None, transform = None, 
            pre_transform = None, pre_filter = None, 
            log = True, force_reload = False
        ):
        """NIST dataset for train, valid and test. NIST_ROOT generated from ./data/nist2data.py

        Args:
            nist_dir (str): path to NIST_ROOT
            inchi_keys (list[str], optional): _description_. Defaults to None, 
                all inchikeys in NIST_ROOT/ contains in the dataset.
            root (_type_, optional): _description_. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            pre_filter (_type_, optional): _description_. Defaults to None.
            log (bool, optional): _description_. Defaults to True.
            force_reload (bool, optional): _description_. Defaults to False.
        """
        self.nist_dir = nist_dir
        if inchi_keys is None:
            self.inchi_keys = os.listdir(self.nist_dir)
        else:
            self.inchi_keys = inchi_keys
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.train = train

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [
            os.path.join(self.nist_dir,inchikey_d,pt_path) 
            for inchikey_d in os.listdir(self.nist_dir)
            for pt_path in os.listdir(os.path.join(self.nist_dir,inchikey_d))
        ]

    def process(self):
        pass

    def len(self):
        return len(self.inchi_keys)

    def get(self, idx) -> SpectrumData|list[SpectrumData]:
        if self.train:
            inchi_key = self.inchi_keys[idx]
            rand_idx = np.random.randint(0,len(os.listdir(os.path.join(self.nist_dir,inchi_key))))
            pt_path = os.listdir(os.path.join(self.nist_dir,inchi_key))[rand_idx]
            pt_path = os.path.join(self.nist_dir,inchi_key,pt_path)
            data = torch.load(pt_path,weights_only=False)
            return data
        else:
            inchi_key = self.inchi_keys[idx]
            pt_path = [
                os.path.join(self.nist_dir,inchi_key,path) for path in 
                os.listdir(os.path.join(self.nist_dir,inchi_key))
            ]
            data = [torch.load(path,weights_only=False) for path in pt_path]
            return data

class NISTDataModule(L.LightningDataModule):
    def __init__(
            self, 
            nist_dir:str = "path/to/dir", 
            train_val_test_split = None,
            train_inchikeys: list[str] = None,
            valid_inchikeys: list[str] = None,
            test_inchikeys: list[str] = None,
            batch_size: int = 32,
            skip_preparation = True,
        ):
        super().__init__()
        self.nist_dir = nist_dir
        self.batch_size = batch_size
        self.skip_preparation = skip_preparation
        if train_val_test_split is not None and os.path.exists(train_val_test_split):
            with open(train_val_test_split,'r') as f:
                split_dict = json.load(f)
                train_inchikeys, valid_inchikeys, test_inchikeys = split_dict['train'], split_dict['valid'], split_dict['test']
        self.train_inchikeys = train_inchikeys
        self.valid_inchikeys = valid_inchikeys
        self.test_inchikeys = test_inchikeys

    def setup(self, stage):
        if stage == "fit":
            self.train_set = NISTDataset(nist_dir=self.nist_dir,inchi_keys=self.train_inchikeys,train=True)
            self.valid_set = NISTDataset(nist_dir=self.nist_dir,inchi_keys=self.valid_inchikeys,train=False)
        if stage == "test":
            self.test_set = NISTDataset(nist_dir=self.nist_dir,inchi_keys=self.test_inchikeys,train=False)

    def prepare_fn(self,spectrum_block:str,loss_formulas):
        spectrum = MSSpectrum.from_NIST_MSP(spectrum_block)
        inchi_key = spectrum.INCHIKEY
        spectrum_dir = os.path.join(self.nist_dir,inchi_key)
        data = spectrum2data(spectrum,loss_formulas=loss_formulas)
        # save data to .pt
        pt_path = os.path.join(spectrum_dir,'{}.pt'.format(uuid.uuid4().hex))
        if data is not None:
            if not os.path.exists(spectrum_dir):
                os.makedirs(spectrum_dir,exist_ok=True)
            with open(pt_path,'wb') as f:
                torch.save(data,f)

    def prepare_data(self, nist_file=None, loss_formulas:list[Formula]=None):
        if self.skip_preparation:
            return
        else:
            work_fn = partial(self.prepare_fn,loss_formulas=loss_formulas)
            with Pool() as pool:
                results = pool.imap(work_fn,NIST_spectrum_iterator(nist_file=nist_file))
                for r in results:
                    pass

    def train_val_test_split(self, train_frac=0.8, val_frac=0.1, test_frac=0.1):
        all_inchikeys = os.listdir(self.nist_dir)
        train, valid, test = random_split(
            all_inchikeys,lengths=[train_frac,val_frac,test_frac],
            generator=torch.Generator().manual_seed(42)    
        )
        return list(train), list(valid), list(test)

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=8,
                          shuffle=True,collate_fn=lambda b:SpectrumDataBatch.from_data_list(b))
    
    def val_dataloader(self):
        return DataLoader(self.valid_set,batch_size=1,num_workers=8,
                          shuffle=False,collate_fn=lambda b:SpectrumDataBatch.from_data_list(b[0]))
    
    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=1,num_workers=8,
                          shuffle=False,collate_fn=lambda b:SpectrumDataBatch.from_data_list(b[0]))

    def transfer_batch_to_device(self, batch:SpectrumDataBatch, device, dataloader_idx):
        batch = batch.to(device)
        return batch

    def train_val_test_stat(self):
        train_count, valid_count, test_count = 0, 0, 0
        for train_inchi_k in self.train_inchikeys:
            train_count += len(os.listdir(os.path.join(self.nist_dir, train_inchi_k)))
        for valid_inchi_k in self.valid_inchikeys:
            valid_count += len(os.listdir(os.path.join(self.nist_dir, valid_inchi_k)))
        for test_inchi_k in self.test_inchikeys:
            test_count += len(os.listdir(os.path.join(self.nist_dir, test_inchi_k)))
        print("Train count: {} spectra, {} mols".format(train_count,len(self.train_inchikeys)))
        print("Valid count: {} spectra, {} mols".format(valid_count,len(self.valid_inchikeys)))
        print("Test count: {} spectra, {} mols".format(test_count,len(self.test_inchikeys)))

############################# Test/Inference Dataset #############################
# Test/Inference data from csv file
# HEAD: Name, SMILES, Precursor_type, NCE, Instrument_type
##################################################################################

class MSMolDataBatch(Batch):
    @classmethod
    def from_input_list(
        cls,
        msmol_list: list[MSMol],
        covariates_list: list[dict],
        loss_formulas:list,
        include_ring_cuts = True,
        num_eigen = 8,
        padding_value = 0.,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        data_list = MSMolDataBatch.add_virtual_and_padding(
            msmol_list=msmol_list,
            covariates_list=covariates_list,
            loss_formulas=loss_formulas,
            include_ring_cuts=include_ring_cuts,
            num_eigen=num_eigen,
            padding_value=padding_value
        )
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)  # type: ignore
        batch._slice_dict = slice_dict  # type: ignore
        batch._inc_dict = inc_dict  # type: ignore

        return batch
    
    @staticmethod
    def add_virtual_and_padding(
        msmol_list: list[MSMol],
        covariates_list: list[dict],
        loss_formulas:list[Formula],
        include_ring_cuts = True,
        num_eigen:int = 8,
        padding_value = 0.,
    ):
        # 1. pad length according to largest frag (i.e. the largest mol iteself as fragment)
        # + 1 to in clude the added virtual node
        pad_length = max([len(msmol.mol_graph) for msmol in msmol_list]) + 1
        # 2. add virtual node to each data graph
        out_data_list = []
        for msmol,covariates_d in zip(msmol_list,covariates_list):
            # n: node, b: bond, e: edge
            fragments = msmol.get_fragments_from_single_cut(include_ring_cuts=include_ring_cuts)
            num_nodes, num_frags = len(msmol.mol_graph), len(fragments)
            n, b, e_i = graph2array(msmol.mol_graph)
            e_vecs, e_vals = graph_laplacian(edge_index=e_i,num_nodes=num_nodes,
                                             num_eigen=num_eigen,padding_value=padding_value)
            
            mz_matrix,formula_matrix = msmol.get_mz_formula_matrix(fragments=fragments,loss_formulas=loss_formulas)
            
            instrument_onehot = [covariates_d['INSTRUMENT_TYPE'] == T for T in INSTRUMENT_TYPES]
            precursor_onehot = [covariates_d['PRECURSOR_TYPE'] == T for T in PRECURSOR_TYPES]
            nce = covariates_d['NCE']
            covariates = np.array([*instrument_onehot,*precursor_onehot,nce],dtype=np.float32).reshape([1,6])

            node_h = np.zeros(shape=(num_nodes+1,*n.shape[1:]),dtype=np.float32)
            node_h[1:,:] = n
            
            node_mask = np.ones(shape=(num_nodes+1,1),dtype=np.float32)
            node_mask[0,:] = 0.
            edge_index = e_i + 1 # first node is virtual node

            eigvecs = np.zeros(shape=(num_nodes+1,num_eigen),dtype=np.float32)
            eigvecs[1:,:] = e_vecs
            # e_vals in shape [1, num_eigen], do not need padding

            frag_node_index = np.zeros(shape=(num_frags,pad_length),dtype=np.int64)
            frag_node_count = np.zeros(shape=(num_frags,1),dtype=np.float32)

            for i, node_index in enumerate([fragment.ion_nodes for fragment in fragments]): 
                frag_node_index[i,:len(node_index)] = [idx+1 for idx in node_index] 
                frag_node_count[i] = float(len(node_index))
            
            # 2.1 eigenvecs
            
            out_data_list.append(
                SpectrumData(
                    node_h = torch.from_numpy(node_h),
                    node_mask = torch.from_numpy(node_mask),
                    edge_index = torch.from_numpy(edge_index),
                    edge_attr = torch.from_numpy(b),
                    covariates = torch.from_numpy(covariates),
                    eigvecs = torch.from_numpy(eigvecs),
                    eigvals = torch.from_numpy(e_vals),
                    frag_node_index = torch.from_numpy(frag_node_index),
                    frag_node_count = torch.from_numpy(frag_node_count),
                    num_frags = num_frags,
                    mz_matrix = torch.from_numpy(mz_matrix),
                    formula_matrix = torch.from_numpy(formula_matrix),
            ))
        return out_data_list
    
    def get_batch_fragment(self,device):
        fragment_batch = []
        for (graph_index,num_frag) in zip(range(self.num_graphs),self.num_frags):
            fragment_batch.extend([graph_index]*int(num_frag))
        return torch.tensor(fragment_batch,dtype=torch.int64,device=device)
