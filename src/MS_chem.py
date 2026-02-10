import rdkit.Chem as Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import networkx as nx
import numpy as np
import itertools, re, copy
from collections import defaultdict, namedtuple
from .definition import *
from typing import *

Fragment = namedtuple('Fragment',['ion_nodes','loss_nodes','frag_formula','ion_charge'])
Peak = namedtuple('Peak',['index','mz','intensity','ppm','ion_formula','is_isotope'])
MASS_ARRAY = np.array([MONO_MASSES[E] for E in ELEMENTS],dtype=np.float32)

def resolve_formula(formula_input) -> defaultdict[str,int]:
    formula_dict = defaultdict(int)
    for ele,i in re.findall(r"(Cl|Br|[HCNOSPFI])(\d*)",formula_input):
        element_count = int(i) if i != '' else 1
        formula_dict[ele] = element_count
    return formula_dict

def combine_fraction(num_ele, num_pick, pick_prob:float) -> float:
    # comput C_num_ele^num_pick
    if num_pick == 0:
        return (1.0 - pick_prob) ** (num_ele-num_pick)
    else:
        combine_count = np.prod(list(range(1,num_ele+1).__reversed__())[:num_pick]).item() / np.prod(list(range(1,num_ele+1))[:num_pick]).item()
        combine_prob = (pick_prob ** num_pick) * (1.0 - pick_prob) ** (num_ele-num_pick)
        return combine_count * combine_prob

class Formula():
    def __init__(self,formula_input=None):
        if isinstance(formula_input,str):
            self.formula_dict = resolve_formula(formula_input)
        elif formula_input is None:
            self.formula_dict = defaultdict(int)
        elif isinstance(formula_input,defaultdict):
            self.formula_dict = formula_input
        else:
            raise NotImplementedError

    @classmethod
    def from_array(cls,formula_array:np.ndarray):
        formula_array = formula_array.flatten()
        formula_dict = defaultdict(int)
        assert formula_array.size == len(ELEMENTS)
        for i, ele in enumerate(ELEMENTS):
            formula_dict[ele] = formula_array[i]
        return cls(formula_input=formula_dict)
        
    @staticmethod
    def convert_input(inputs):
        if isinstance(inputs,str):
            return Formula(inputs)
        else:
            assert isinstance(inputs,Formula)
            return inputs
    
    @property
    def mass(self) -> float:
        mass = 0.
        for ele, i in self.formula_dict.items():
            mass += MONO_MASSES[ele] * i
        return mass
    
    def have_isotope(self) -> bool:
        return any([self.formula_dict[ele] >0 for ele in ELEMENTS_ISO])
    
    def contain_subformula(self,subformula) -> bool:
        subformula = self.convert_input(subformula)
        return all([
            self.formula_dict[ele] >= count for ele,count in subformula.formula_dict.items()
        ])
      
    def __add__(self,adduct):
        adduct = self.convert_input(adduct)
        formula_dict = copy.deepcopy(self.formula_dict)
        for ele,count in adduct.formula_dict.items():
            if count == 0:
                continue
            else:
                formula_dict[ele] += count
        return Formula(formula_dict)

    def __sub__(self,loss):
        loss = self.convert_input(loss)
        formula_dict = copy.deepcopy(self.formula_dict)
        for ele,count in loss.formula_dict.items():
            if count == 0:
                continue
            else:
                formula_dict[ele] -= count
        if any([count < 0 for count in formula_dict.values()]):
            return Formula()
        else:
            return Formula(formula_dict)

    def __mul__(self,f:int):
        formula_dict = copy.deepcopy(self.formula_dict)
        for ele in self.formula_dict.keys():
            formula_dict[ele] = int(f*formula_dict[ele])
        return Formula(formula_dict)

    def __eq__(self,inputs):
        inputs = self.convert_input(inputs)
        return all([self.formula_dict[ele] == inputs.formula_dict[ele] for ele in ELEMENTS])
    
    def is_empty(self):
        return all([self.formula_dict[ele]==0 for ele in self.formula_dict.keys()])

    def is_neutral(self) -> bool:
        # 1. N rule, odd N and P, odd H + halogen
        H_count = sum([self.formula_dict[ele] for ele in ['H','F','Cl','Br','I']])
        N_count = self.formula_dict['N'] + self.formula_dict['P']
        if ((H_count%2 == 0) ^ (N_count%2 == 0)) and H_count > 0: # NO, NO2 
            return False
        # 2. highest valency
        heavy_count = self.formula_dict['C'] + self.formula_dict['N'] + self.formula_dict['P']
        if heavy_count == 0: # H2O, 
            return True
        else:
            H_max = 2 * self.formula_dict['C'] + self.formula_dict['N'] + self.formula_dict['P'] + 2
            return H_count <= H_max

    def isotope_mass_distribution(self) -> tuple[np.ndarray,np.ndarray]:
        cl_count, br_count = self.formula_dict['Cl'],self.formula_dict['Br']
        if cl_count > 0:
            cl_distribution = np.array([combine_fraction(cl_count,cl_iso_count,CL37_PER) for cl_iso_count in range(cl_count+1)],dtype=np.float32)
            cl_mass = np.array([(CL37_MASS-MONO_MASSES['Cl'])*cl_iso_count for cl_iso_count in range(cl_count+1)],dtype=np.float32)
        else:
            cl_distribution = np.ones(shape=(1,),dtype=np.float32)
            cl_mass = np.zeros(shape=(1,),dtype=np.float32)
        if br_count > 0:
            br_distribution = np.array([combine_fraction(br_count,br_iso_count,BR81_PER) for br_iso_count in range(br_count+1)],dtype=np.float32)
            br_mass = np.array([(BR81_MASS-MONO_MASSES['Br'])*br_iso_count for br_iso_count in range(cl_count+1)],dtype=np.float32)
        else:
            br_distribution = np.ones(shape=(1,),dtype=np.float32)
            br_mass = np.zeros(shape=(1,),dtype=np.float32)
        
        prob_distribution = cl_distribution[:,np.newaxis] * br_distribution[np.newaxis,:]
        mass = cl_mass[:,np.newaxis] + br_mass[np.newaxis,:]
        return prob_distribution.flatten(), mass.flatten() + self.mass

    def to_array(self):
        return np.array([self.formula_dict[E] for E in ELEMENTS],dtype=np.int16)

    def __str__(self):
        out_formula = ''
        for ele in ELEMENTS:
            if self.formula_dict[ele] == 1:
                out_formula += ele
            elif self.formula_dict[ele] > 1:
                out_formula += ele + str(int(self.formula_dict[ele]))
            else:
                continue
        return out_formula

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return "Formula: {}".format(str(self))

def mol2nx(mol:Chem.rdchem.Mol) -> nx.Graph:
    """Initialized mol graph from Mol,

    Args:
        mol (Chem.rdchem.Mol): input Mol

    Returns:
        nx.Graph: Mol Graph, add
    """
    G = nx.Graph()
    for i in range(mol.GetNumAtoms()):    
        if str(mol.GetAtomWithIdx(i).GetHybridization()) in HYBRIDIZATION:
            hybridization = str(mol.GetAtomWithIdx(i).GetHybridization())
        else:
            hybridization = 'OTHERS'
        G.add_node(
            i,
            element=mol.GetAtomWithIdx(i).GetSymbol(),
            charge=mol.GetAtomWithIdx(i).GetFormalCharge(),
            hybridization=hybridization,
            chiral=str(mol.GetAtomWithIdx(i).GetChiralTag()),
            Hs=mol.GetAtomWithIdx(i).GetNumExplicitHs() + mol.GetAtomWithIdx(i).GetNumImplicitHs()
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            # bond_type:    Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
            #               Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            bond_type=str(bond.GetBondType()),
            in_ring=int(bond.IsInRing()),
        )
    return G

def graph2formula(mol_graph:nx.Graph) -> Formula:
    F = Formula()
    for i, atom_prop in mol_graph.nodes(data=True):
        F = F + atom_prop['element']
        if atom_prop['Hs'] > 0:
            F = F + 'H{:d}'.format(atom_prop['Hs'])
    return F

##### functions for detecting rings in mol 
def get_ring_cuts(G:nx.Graph):
    """Get set of cuts (two edges in a ring), resulted into two conponents

    Args:
        G (nx.Graph): input mol graph

    Returns:
        _type_: set[tuple[tuple[int]]], each element in set, e.g. ((0,1),(2,3)) two edges combination
            lead into two conponents of the full ring
    """
    single, fused, bridged = get_merge_rings(G)
    ring_cuts = set()
    for ring in single:
        ring_graph:nx.Graph = G.subgraph(ring)  
        # isolated single ring, simply enumerate combination of two edges
        ring_cuts.update(itertools.combinations(ring_graph.edges,2))
    for ring in fused:
        ring_graph:nx.Graph = G.subgraph(ring)
        cut_edges = []
        for (i,j) in ring_graph.edges:
            if len(list(ring_graph.neighbors(i))) == 3 and len(list(ring_graph.neighbors(j))) == 3:
                continue
            else:
                cut_edges.append((i,j))
        ring_cuts.update(itertools.combinations(cut_edges,2))
    for ring in bridged:
        ring_graph:nx.Graph = G.subgraph(ring)
        # enumeration and test connectivity
        for cut in itertools.combinations(ring_graph.edges,2):
            tmp_graph = nx.Graph(ring_graph)
            tmp_graph.remove_edges_from(cut)
            if len(list(nx.connected_components(tmp_graph))) == 2:
                ring_cuts.add(cut)
    return ring_cuts

def get_merge_rings(G:nx.Graph):
    ring_nodes = [set(ring) for ring in nx.cycle_basis(G)]
    if len(ring_nodes) <= 1:
        return ring_nodes,[],[]
    else:
        single,fused,bridged = _merge_rings_helper(ring_nodes,is_single=[True for _ in range(len(ring_nodes))],
                                                    single=[],fused=[],bridged=[])
        return single,merge_fuse_ring(fused),bridged

def _merge_rings_helper(
        unclassed:list[set],
        is_single:list[bool],
        single:list,
        fused:list,
        bridged:list
    ):
    if len(unclassed) == 0:
        return single,fused,bridged
    else:
        ring_nodes_1 = unclassed[0]
        for i,ring_nodes_2 in enumerate(unclassed[1:],1):
            if len(ring_nodes_1 & ring_nodes_2) == 1:
                # shared nodes = 1, spiro rings; do not merge these two rings
                continue
            elif len(ring_nodes_1 & ring_nodes_2) == 2:
                # shared nodes = 2, fused rings;
                is_single[0] = False
                is_single[i] = False
                fused.append(ring_nodes_1|ring_nodes_2)
            elif len(ring_nodes_1 & ring_nodes_2) > 2:
                # shared nodes > 2, brideged rings;
                is_single[0] = False
                is_single[i] = False
                bridged.append(ring_nodes_1|ring_nodes_2)
        if is_single[0]:
            single.append(ring_nodes_1)
        return _merge_rings_helper(unclassed[1:],is_single[1:],single,fused,bridged)

def merge_fuse_ring(fused:list[set]):
    # _merge_ring_helper only get fused rings with two simple ring
    # fused ring can made up by several rings 
    if len(fused) <= 1:
        return fused
    else:
        finish = False
        while not finish:
            merge_complete = True
            for ring_i,ring_j in itertools.combinations(fused,2):
                if len(ring_i&ring_j) >= 2:
                    fused.remove(ring_i)
                    fused.remove(ring_j)
                    fused.append(ring_i|ring_j)
                    merge_complete = False
                    break
            if merge_complete:
                finish = True
        return fused
###### end functions for ring detection

def resolve_precursor(precursor_input) -> float:
    # only consider precursor_input [M+H]+ [M-H]-/ [M+H+2i]+
    precursor_match = re.match(r'\[(M[\+\-]H)\+?(\d?)i?\][\+\-]',precursor_input)
    if precursor_match.group(1) == 'M+H':
        free_op = 1
    elif precursor_match.group(1) == 'M-H':
        free_op = -1
    else:
        raise NotImplementedError
    if precursor_match.group(2) == '':
        neutron_count = 0
    else:
        neutron_count = int(precursor_match.group(2))
    return free_op, neutron_count

class MSMol():
    def __init__(
        self,
        mol: Chem.rdchem.Mol,
        precursor_input: str,
    ):
        self.mol = mol
        self.mol_graph = mol2nx(mol)
        self.mol_formula = Formula(CalcMolFormula(mol))
        self.free_op,_ = resolve_precursor(precursor_input)
        self.precursor_formula:Formula = self.mol_formula + (Formula('H') * self.free_op)
        self.ion_charge = self.free_op
    
    def get_fragments_from_single_cut(self,include_ring_cuts=True) -> list[Fragment]: 
        fragments = [] # outputs
        for (i,j) in self.mol_graph.edges:
            graph = nx.Graph(self.mol_graph)
            graph.remove_edge(i,j)
            ion_nodes_frag = list(nx.connected_components(graph))
            if len(ion_nodes_frag) == 1:
                # i,j is a ring bond, deal with it afterwards
                continue
            else:
                # each frag can be ion, the remained are neutral loss
                # two types of cleavages:
                # type1: CH3CH2-CH2CH3/+H+ -> CH2CH2/+H+ and CH3CH3, the ion will loss an H
                # type2: CH3CH2-CH2CH3/+H+ -> CH3CH3/+H+ and CH2CH2, the ion will get an H 
                for ion_nodes in ion_nodes_frag:
                    loss_nodes = [i for i in self.mol_graph if i not in ion_nodes]
                    frag_formula = self.get_subformula_from_nodes(ion_nodes)
                    fragments.append(Fragment(ion_nodes=tuple(sorted(ion_nodes)),loss_nodes=tuple(loss_nodes),
                                              frag_formula=frag_formula,ion_charge=self.ion_charge))
        # the full molecule, i.e. precursor
        fragments.append(
            Fragment(
                ion_nodes=tuple(sorted(list(self.mol_graph))),loss_nodes=tuple(),
                frag_formula=self.mol_formula,ion_charge=self.ion_charge
            ))
        if include_ring_cuts:
            ring_cuts = get_ring_cuts(self.mol_graph)
            for cut_edges in ring_cuts:
                graph = nx.Graph(self.mol_graph)
                graph.remove_edges_from(cut_edges)
                ion_nodes_frag = list(nx.connected_components(graph))
                if len(ion_nodes_frag) == 1: # for safe
                    continue
                else:
                    for ion_nodes in ion_nodes_frag:
                        loss_nodes = [i for i in self.mol_graph if i not in ion_nodes]
                        frag_formula = self.get_subformula_from_nodes(ion_nodes)
                        fragments.append(Fragment(ion_nodes=tuple(sorted(ion_nodes)),loss_nodes=tuple(loss_nodes),
                                                frag_formula=frag_formula,ion_charge=self.ion_charge))                    
        else: # for testing FIORA
            pass
        # remove duplicated fragments
        out_fragments:list[Fragment] = []
        nodes_set = set()
        for fragment in fragments:
            if fragment.ion_nodes not in nodes_set:
                out_fragments.append(fragment)
                nodes_set.add(fragment.ion_nodes)
            else:
                continue
        out_fragments = [fragment for fragment in out_fragments if len(fragment.ion_nodes) > 1]
        out_fragments = sorted(out_fragments,key = lambda frag:len(frag.ion_nodes))
        return out_fragments

    def get_subformula_from_nodes(self,nodes:list[int]) -> Formula:
        F = Formula()
        for node in nodes:
            F = F + self.mol_graph.nodes[node]['element']
            Hs = self.mol_graph.nodes[node]['Hs']
            if Hs > 0:
                F = F + 'H{:d}'.format(Hs)
        return F
    
    def get_mz_formula_matrix_old(self,fragments:list[Fragment],loss_formulas:list[Formula]) -> tuple[np.ndarray, np.ndarray]:
        mzs,formulas_matrix = [],[]
        for fragment in fragments:
            base_formula:Formula = self.get_subformula_from_nodes(fragment.ion_nodes) + (Formula('H') * self.free_op)
            if base_formula != Formula(''):
                level_0H = [base_formula - loss_f for loss_f in loss_formulas]
                level_add1H = [base_formula + Formula('H') - loss_f for loss_f in loss_formulas]
                level_min1H = [base_formula - Formula('H') - loss_f for loss_f in loss_formulas]
                mzs.append([
                    [f.mass for f in level_0H],
                    [f.mass for f in level_add1H],
                    [f.mass for f in level_min1H],
                ])
                formulas_matrix.append([
                    [[f.formula_dict[ele] for ele in ELEMENTS] for f in level_0H],
                    [[f.formula_dict[ele] for ele in ELEMENTS] for f in level_add1H],
                    [[f.formula_dict[ele] for ele in ELEMENTS] for f in level_min1H],
                ])
            else:
                mzs.append([
                    [0. for _ in range(len(loss_formulas))],
                    [0. for _ in range(len(loss_formulas))],
                    [0. for _ in range(len(loss_formulas))]
                ])
                formulas_matrix.append([
                    [[0 for _ in ELEMENTS] for _ in level_0H],
                    [[0 for _ in ELEMENTS] for _ in level_add1H],
                    [[0 for _ in ELEMENTS] for _ in level_min1H],
                ])
        mzs, formulas_matrix = np.array(mzs,dtype=np.float32), np.array(formulas_matrix,dtype=np.int16)
        formula_mask = np.any(formulas_matrix < 0,axis=3,keepdims=True)
        mzs = mzs - (self.ion_charge * ELECTRON_MASS)   
        mzs = np.where(np.squeeze(formula_mask,axis=3),0.,mzs)
        mzs[-1,1:,:] = 0. # the last fragment do not include +1H -1H, TEST
        formulas_matrix[-1,1:,:,:] = 0 # the last fragment do not include +1H -1H, TEST
        return mzs, formulas_matrix   

    def get_mz_formula_matrix(self,fragments:list[Fragment],loss_formulas:list[Formula]) -> tuple[np.ndarray, np.ndarray]:       
        # loss_formulas_array in shape [num_loss, num_elements]  
        base_formulas = [self.get_subformula_from_nodes(fragment.ion_nodes) + (Formula('H') * self.free_op) 
                         for fragment in fragments]
        formula_matrix = np.stack([formula.to_array() for formula in base_formulas]) # shape [num_frag, num_ele]
        formula_matrix = np.concatenate([
            formula_matrix[:,np.newaxis,:],
            formula_matrix[:,np.newaxis,:] + Formula('H').to_array().reshape([1,1,-1]),
            formula_matrix[:,np.newaxis,:] - Formula('H').to_array().reshape([1,1,-1]),
        ],axis = 1) # shape [num_frag,3,num_ele]
        loss_formulas_array = np.stack([loss_formula.to_array() for loss_formula in loss_formulas])
        formula_matrix = formula_matrix[:,:,np.newaxis,:] - loss_formulas_array[np.newaxis,np.newaxis,:,:] 
        # now formula_matrix in shape shape [num_frag,3,nu,_loss,num_ele]
        formula_mask = np.any(formula_matrix < 0,axis=3,keepdims=True) 
        formula_matrix = np.where(formula_mask, np.zeros_like(formula_matrix), formula_matrix)
        # mzs
        mzs = np.sum(formula_matrix * MASS_ARRAY.reshape([1,1,1,-1]),axis=-1)
        mzs = mzs - (self.ion_charge * ELECTRON_MASS)
        mzs = np.where(np.squeeze(formula_mask,axis=3),0., mzs)
        mzs[-1,1:,:] = 0. # the last fragment do not include +1H -1H, TEST
        formula_matrix[-1,1:,:,:] = 0 # the last fragment do not include +1H -1H, TEST
        return mzs, formula_matrix  
   
class MSSpectrum():
    def __init__(
        self,
        precursor: str,
        peaks: list[Peak],
        nist_index = None,
        **kwargs
    ):
        self.precursor = precursor
        self.raw_peaks = peaks
        self.ppm_cutoff = 20.0
        self.da_cutoff = 0.01
        self.nist_index = nist_index # only used for index nist spectra, default None
        for prop, value in kwargs.items():
            self.__setattr__(prop,value)

    @classmethod
    def from_NIST_MSP(cls,msp_block:str):
        property_dict = {}
        nist_index = None
        mz_diff = 0.0
        info_lines, peak_lines = [], []
        peak_pattern = re.compile(r"^[\d]+")
        # split according to ':'
        # info_pattern = re.compile(r"([\w\s_#\-]+): ([\[\]\w\s_\+\-,.()]+)") 
        info_pattern = re.compile(r"([\w\s_#\-]+): ([\[\]\w\W]+)") 
        for line in msp_block.split('\n'): 
            if re.match(peak_pattern,line) is not None:
                peak_lines.append(line)
            else:
                info_lines.append(line)
        for line in info_lines:
            info_match = re.match(info_pattern,line)
            if info_match is not None:
                if info_match.group(1) == 'Notes':
                    mz_diff_match = re.match(r".*Mz_diff=(-?[\d\.]+)ppm.*",info_match.group(2))
                    if mz_diff_match is not None:
                        mz_diff = float(mz_diff_match.group(1))
                elif info_match.group(1) == 'NISTNO':
                    nist_index = info_match.group(2) # only used for index nist spectra
                elif info_match.group(1).upper() in NIST_MSP_KEYWORDS:
                    property_dict[info_match.group(1).upper()] = info_match.group(2)
        peaks = [cls.resolve_msp_peak_line(i,peak_line,mz_diff=mz_diff) for (i,peak_line) in enumerate(peak_lines)]
        property_dict['BLOCK'] = msp_block # the associated full text in msp file
        #precursor_sub = re.sub(r'\+\d?i','',property_dict['PRECURSOR_TYPE'])
        if property_dict['PRECURSOR_TYPE'] in ['[M+H]+','[M-H]-']:
            return cls(peaks=peaks,precursor=property_dict['PRECURSOR_TYPE'],nist_index=nist_index,**property_dict)
        else:
            print('Unspoorted Precursor: {}'.format(property_dict['PRECURSOR_TYPE']))
            return None

    @staticmethod
    def resolve_msp_peak_line(peak_index:int,peak_line:str,mz_diff=0.0) -> Peak:
        # eg. 97.0284 2.20 "C5H5O2=p-C16H20O8/0.0ppm 11/29"
        # 115.0867 1.80 "C5H11N2O=p-C11H11N3O2/1.0ppm;C10H22N4O2^2=p-C6HNO^2/-1.4ppm 23/29"
        # 1. split according to blankspace
        peak_line = re.sub(r'"','',peak_line)
        peak_contents = peak_line.strip().split(maxsplit=2)
        if len(peak_contents) == 3:
            mz, intensity, annotatation = peak_contents
        elif len(peak_contents) == 2:
            mz, intensity = peak_contents
            return Peak(
                        index = peak_index, mz = float(mz), intensity = float(intensity), ppm = 0.,
                        ion_formula = Formula(), is_isotope = False
                    )
        else:
            mz, intensity = peak_contents[:2]
            return Peak(
                        index = peak_index, mz = float(mz), intensity = float(intensity), ppm = 0.,
                        ion_formula = Formula(), is_isotope = False
                    )
        # 2. parse annotatation
        annotatation = re.sub(r' ?\d+/\d+$','',annotatation) # strip peak counts
        # most sophisticated anno_pattern, eg. C15H18N2O3+N2+i^2=p-CH5N3+N2+i^2/0.3ppm
        anno_pattern = re.compile(r'''
                                  ([\w]+)           # group 1: match the ion formula, C15H18N2O3
                                  (\+?[\w]*\+?i?)   # group 2: match the adduct/isotope, +N2+i
                                  (\^?\d?)          # group 3: match the charge, ^2
                                  =p-
                                  ([\w\s]+)         # group 4: match the loss formula, CH5N3
                                  (\+?[\w]*\+?i?)   # group 5: match the adduct/isotope, +N2+i
                                  (\^?\d?)/         # group 6: match the charge, ^2
                                  ([\+\-\d.]+)ppm   # group 7: match the ppm, 0.3
                                  ''',re.VERBOSE)        
        # 2.1 possible formula, seperated by ";"
        # gather all possible formulas and choose the one with lowest ppm
        possible_peaks = []
        for anno in annotatation.split(';'):
            # filter pattern
            a_match = re.match(anno_pattern,anno)
            if a_match is None: 
                # maybe the precursor peak, eg. "p/-0.3ppm", "p+i/8.1ppm"
                if re.match(r'p\+?\d?i?/[\+\-\d.]+ppm',anno) is not None:
                    possible_peaks.append(
                        Peak(
                            index = peak_index, mz = float(mz), intensity = float(intensity), ppm = 0.,
                            ion_formula = 'PRECURSOR', is_isotope = False
                    ))
                else: # not a usual annotation, skip annotation
                    possible_peaks.append(
                        Peak(
                            index = peak_index, mz = float(mz), intensity = float(intensity), ppm = 0.,
                            ion_formula = Formula(), is_isotope = False
                    ))
            else:
                ion_str, adduct_str, charge_str, adduct_loss_str, ppm = \
                    a_match.group(1), a_match.group(2), a_match.group(3), a_match.group(5), a_match.group(7)
                # do not return fragments with charge more than 1 eg. C10H22N4O2^2=p-C6HNO^2/-1.4ppm
                if charge_str != '':
                    continue
                else:
                    # in some cases, adduct_str is the same as adduct_loss_str:
                    if adduct_str == adduct_loss_str:
                    # parse adduct_str +N2+i
                        is_isotope= False
                        adducts = re.split(r'\+',adduct_str)
                        for adduct in adducts:
                            if adduct == 'i':
                                is_isotope = True
                            else:
                                continue
                        ion_formula = Formula(ion_str)
                        possible_peaks.append(
                            Peak(
                                index = peak_index, mz = float(mz), intensity = float(intensity), ppm = float(ppm),
                                ion_formula = ion_formula, is_isotope = is_isotope
                        ))
                    else:
                        # eg. C6H5N2+i=p-CH3NO2+N2+i
                        is_isotope= False
                        adducts = re.split(r'\+',adduct_loss_str)
                        adduct_formula = Formula()
                        for adduct in adducts:
                            if adduct == '':
                                continue
                            elif adduct == 'i':
                                is_isotope = True
                            else:
                                adduct_formula = adduct_formula + adduct
                        ion_formula = Formula(ion_str)
                        ion_formula = ion_formula - adduct_formula 
                        # substract the adduct from ion formula
                        # ion_formula only used to map the molecule
                        possible_peaks.append(
                            Peak(
                                index = peak_index, mz = float(mz), intensity = float(intensity), ppm = float(ppm),
                                ion_formula = ion_formula, is_isotope = is_isotope
                        ))
        if len(possible_peaks) > 0:
            return min(possible_peaks,key=lambda peak:abs(peak.ppm-mz_diff))
        else:
            return  Peak(
                        index = peak_index, mz = float(mz), intensity = float(intensity), ppm = 0.,
                        ion_formula = Formula(), is_isotope = False
                    )
        
    @classmethod
    def from_MSNLIB_MGF(cls,mgf_block:str):
        property_dict = {}
        info_lines, peak_lines = [], []
        peak_pattern = re.compile(r"^[\d]+")
        info_pattern = re.compile(r"([\w\W_]+)=([\[\]\w\W=_#\-]+)") 
        for line in mgf_block.split('\n'): 
            if re.match(peak_pattern,line) is not None:
                peak_lines.append(line)
            else:
                info_lines.append(line)
        peaks = [cls.resolve_mgf_peak_line(i,peak_line) for (i,peak_line) in enumerate(peak_lines)]   
        for line in info_lines:   
            info_match = re.match(info_pattern,line)
            if info_match is not None and info_match.group(1).upper() in MSNLIB_MGF_KEYWORDS:
                property_dict[info_match.group(1).upper()] = info_match.group(2)
        property_dict['BLOCK'] = mgf_block
        property_dict['PRECURSOR_TYPE'] = property_dict.pop('ADDUCT')
        property_dict['COLLISION_ENERGY'] = property_dict.pop('COLLISION_ENERGY').strip('[]')
        property_dict['INSTRUMENT_TYPE'] = property_dict.pop('FRAGMENTATION_METHOD')
        if property_dict['PRECURSOR_TYPE'] in ['[M+H]+','[M-H]-']:
            return cls(peaks=peaks,precursor=property_dict['PRECURSOR_TYPE'],**property_dict)
        else:
            print('Unspoorted Precursor: {}'.format(property_dict['PRECURSOR_TYPE']))
            return None

    @staticmethod
    def resolve_mgf_peak_line(peak_index:int,peak_line:str) -> Peak:
        # mgf peak line do not have annotations: eg. 118.065125 3.647
        # simply split and record mz intensity
        mz, intensity = map(float,peak_line.split())
        return Peak(
            index = peak_index, mz = mz, intensity = intensity, ppm = 0.,
            ion_formula = Formula(), is_isotope = False
        )

    def get_mol(self) -> Chem.rdchem.Mol|None:
        if hasattr(self,'SMILES'):
            mol = Chem.MolFromSmiles(self.SMILES)
        elif hasattr(self,'INCHIKEY'):
            try:
                mol_inchi = INCHIKEY_TABLE[self.INCHIKEY]
                mol =  Chem.MolFromInchi(mol_inchi)
            except KeyError:
                return None
        else:
            return None
        atom_elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if any([e not in ELEMENTS for e in atom_elements]):
            return None
        else:
            return mol
    
    def get_nce(self) -> float:
        # normalize collision energy
        # nist COLLISION_ENERGY have two types:
        # e.g. Collision_energy: 30, spectrum.COLLISION_ENERGY = '30.0'
        #      Collision_energy: NCE=25% 9eV, spectrum.COLLISION_ENERGY = 'NCE=25% 9eV'
        #      Collision_energy: NCE=35%, spectrum.COLLISION_ENERGY = 'NCE=35%'
        energy_match = re.match(r"NCE=(\d+)% ?(\d*)e?V?",self.COLLISION_ENERGY)
        if energy_match is None:
            precursor_mass = MSMol(self.get_mol(),self.precursor).precursor_formula.mass
            # equation from Bioinformatics, 2023, 39(6), btad354 https://doi.org/10.1093/bioinformatics/btad354
            # 3DMolMS: prediction of tandem mass spectra from 3D molecular conformations
            nce = (float(self.COLLISION_ENERGY) * 500.0) / (precursor_mass * 100.0)
        else:
            nce = float(energy_match.group(1)) / 100.0
        return nce
    
    def get_collision_energy(self) -> float:
        energy_match = re.match(r"NCE=(\d+)% ?(\d*)e?V?",self.COLLISION_ENERGY)
        if energy_match is None:
            collision_energy = float(self.COLLISION_ENERGY)
        else:
            if energy_match.group(2) != '':
                collision_energy = float(energy_match.group(2))
            else:
                precursor_mass = MSMol(self.get_mol(),self.precursor).precursor_formula.mass
                nce = float(energy_match.group(1)) / 100.0
                collision_energy = nce * precursor_mass / 5.0
        return collision_energy

    def merge_peaks(self) -> list[Peak]:
        msmol = MSMol(self.get_mol(),self.precursor)
        formula_group = defaultdict(list) # key: formula_str, value: list[Peak]
        unknown_peaks:list[Peak] = [] # peaks with unknown formula
        for peak in self.raw_peaks:
            if str(peak.ion_formula) == '': # do not know formula 
                if peak.mz < msmol.precursor_formula.mass: # unknown peak with lower mz than precursor
                    unknown_peaks.append(peak)
                else:
                    continue # do not record unknown peak with higher mz than precursor
            elif str(peak.ion_formula) == 'PRECURSOR':
                ion_str = str(msmol.precursor_formula) 
                formula_group[ion_str].append(peak)
            else: # a common peak with confident formula
                ion_str = str(peak.ion_formula)
                formula_group[ion_str].append(peak)
        
        # 1.2 merge mz_peak into formula_peak
        # eg. 57.0131 810.29 "C3H2F=p-C4H4/-7.0ppm 19/19"   
        #     57.0148 8.89 "? 7/19"
        #     merge peak 2 into peak 1
        # criterion: (1). mz peak with unknown formula;
        #            (2). delta mz < self.da_cutoff/self.ppm_cutoff
        for peak in unknown_peaks:
            for ion_str in formula_group.keys():
                ion_mass = Formula(ion_str).mass - (msmol.ion_charge * ELECTRON_MASS)
                delta_mz = abs(peak.mz-ion_mass)
                if delta_mz < self.da_cutoff or delta_mz/ion_mass < self.ppm_cutoff * 1e-6:
                    formula_group[ion_str].append(peak)
                    unknown_peaks.remove(peak)
                    break
            continue
        # 1.3 merge peaks according to formula_group
        merged_peaks = []
        for ion_str, peaks in formula_group.items():
            ion_mass = Formula(ion_str).mass - (msmol.ion_charge * ELECTRON_MASS)
            intensity = sum([peak.intensity for peak in peaks])
            merged_peaks.append(
                Peak(index=len(merged_peaks),mz=ion_mass,intensity=intensity,ppm=0.,ion_formula=Formula(ion_str),is_isotope=False)
            )
        for peak in unknown_peaks:
            merged_peaks.append(
                Peak(index=len(merged_peaks),mz=peak.mz,intensity=peak.intensity,ppm=0.,ion_formula=Formula(),is_isotope=False)
            )
        return merged_peaks

    def get_small_loss(self,include_ring_cuts=True) -> dict[str,float]:
        msmol = MSMol(self.get_mol(),self.precursor)
        merged_peaks = self.merge_peaks()
        intensity_normalization_factor = sum([peak.intensity for peak in merged_peaks])
        fragments = msmol.get_fragments_from_single_cut(include_ring_cuts=include_ring_cuts)
        peaks_with_formula = [peak for peak in merged_peaks if str(peak.ion_formula) != '']
        # match fragments with peaks
        matched_peak_index = set()
        matched_formulas:set[Formula] = set()
        for fragment in fragments:
            # fragment ion can +1H or -1H, then add/sub H according to [M+H]+/[M-H]-
            # fragment ion can also be 0H change, twice cut
            # do not care about frags without H (-H will result into Formula(), see in Formula.__sub__) 
            fragment_ion_formula = fragment.frag_formula + (Formula('H') * msmol.free_op)
            formula_add_1H, formula_min_1H = fragment_ion_formula + Formula('H'),  fragment_ion_formula - Formula('H')
            for i,peak in enumerate(peaks_with_formula):
                if peak.ion_formula == fragment_ion_formula:
                    matched_peak_index.add(i)
                    matched_formulas.add(fragment_ion_formula)
                if peak.ion_formula == formula_add_1H: 
                    matched_peak_index.add(i)
                    matched_formulas.add(formula_add_1H)
                if peak.ion_formula == formula_min_1H:
                    matched_peak_index.add(i)
                    matched_formulas.add(formula_min_1H)            
        unmatched_peaks:list[Peak] = [peaks_with_formula[i] for i in range(len(peaks_with_formula)) if i not in matched_peak_index]
        if len(unmatched_peaks) == 0:
            return dict()
        else:
            loss_formulas = defaultdict(float)
            for (m_formula,u_peak) in itertools.product(matched_formulas,unmatched_peaks):
                # assume unmatch_peak is result from matched_peak by dropping a small frag
                # check whether matched_peak 'contains' unmatched_peak
                if m_formula.contain_subformula(u_peak.ion_formula):
                    loss = m_formula - u_peak.ion_formula
                    loss_formulas[loss] += u_peak.intensity / intensity_normalization_factor
                else:
                    continue
            # different spectrums contains different number of peaks, loss_formulas need to be weighted by matched_formulas 
            normalization = len(matched_formulas) 
            loss_formulas = {str(formula):weight/normalization for (formula,weight) in loss_formulas.items()}
            return loss_formulas
    
    def link_mol(self,loss_formulas:list[Formula],include_ring_cuts=True):
        msmol = MSMol(self.get_mol(),self.precursor)
        # 1. map each fragment (from single cut) to peaks
        merged_peaks = self.merge_peaks()
        peak_formula_matrix = np.array([[peak.ion_formula.formula_dict[ele] for ele in ELEMENTS] for peak in merged_peaks],dtype=np.int16)
        peak_mz_matrix = np.array([peak.mz for peak in merged_peaks],dtype=np.float32)
        # peak_mz_matrix in shape [num_peak], peak_formula_matrix in shape [num_peak, num_ele]

        msmol_fragments = msmol.get_fragments_from_single_cut(include_ring_cuts=include_ring_cuts)
        mz_matrix, formula_matrix = msmol.get_mz_formula_matrix(msmol_fragments,loss_formulas)
        # mz_matrix in shape [num_frag, 3, num_loss_formulas], formula_matrix in shape [num_frag, 3, num_loss_formulas, num_ele]
        nonempty_formula = np.where(np.any(formula_matrix > 0, axis=3),True, False)

        # 0 for unmatch, peak index starts from 1
        fragments_map = np.zeros(shape=(len(msmol_fragments),3,len(loss_formulas)),dtype=np.int16)
        for (peak_index,(peak_mz,peak_formula)) in enumerate(zip(peak_mz_matrix, peak_formula_matrix),1): # peak 0, virtual
            if np.sum(peak_formula) > 0:
                is_match = np.all((formula_matrix - peak_formula)==0, axis=3) & nonempty_formula
            else:
                mz_match_DA = np.where(np.abs(mz_matrix - peak_mz) < self.da_cutoff, True, False)
                mz_match_ppm = np.where(np.abs(mz_matrix - peak_mz) / (mz_matrix + 1e-12) < self.ppm_cutoff * 1e-6, True, False)
                is_match = np.any(np.stack([mz_match_DA,mz_match_ppm],axis=-1),axis=3) 

            fragments_map[is_match] = peak_index            
        
        fragments_map[-1,1:,:] = 0
        return merged_peaks, fragments_map, mz_matrix, formula_matrix

    def prepare_data_for_training(self,loss_formulas:list[Formula],include_ring_cuts=True):
        merged_peaks, fragments_map, mz_matrix, formula_matrix = self.link_mol(loss_formulas,include_ring_cuts=include_ring_cuts)
        # renumber peak index (drop unmatched peaks)
        matched_peaks_index = np.unique(fragments_map)
        matched_peaks = [peak for peak in merged_peaks if (peak.index + 1) in matched_peaks_index]
        # index update for fragments map 
        peak_index_update = {0:0} # key: old_index, new_index
        for (new_idx,peak) in enumerate(matched_peaks,1): # start from 1
            peak_index_update[peak.index+1] = new_idx
        for (old_peak_index,new_peak_index) in peak_index_update.items():
            if old_peak_index != new_peak_index:
                fragments_map = np.where(fragments_map==old_peak_index,new_peak_index,fragments_map)
        # peak intensity
        peak_intensity = [peak.intensity for peak in matched_peaks]
        peak_intensity.insert(0,0.0) # real peak start from 1
        peak_intensity = np.array(peak_intensity,dtype=np.float32).reshape([-1,1])
        if np.sum(peak_intensity) > 0.: # some spectrums have no peaks after filtering
            peak_intensity = peak_intensity / np.sum(peak_intensity) # normalize to sum of 1.0
            return peak_intensity, fragments_map, mz_matrix, formula_matrix
        else:
            return None, None, None, None

if __name__ == '__main__':
    #free_op, neutron_count = resolve_precursor('[M-H+2i]-')
    #print(free_op, neutron_count)
    msp_file = r'E:\work\MS\data\tmp\test.msp'
    with open(msp_file,'r') as f:
        block = f.read()
    #print(block)
    spectrum = MSSpectrum.from_NIST_MSP(block)
    loss_formulas = [Formula(L) for L in LOSS_FORMULAS]
    #mol = spectrum.get_mol()
    peak_intensity, fragments_map = spectrum.prepare_data_for_training(loss_formulas=loss_formulas)
    msmol = MSMol(spectrum.get_mol(),spectrum.precursor)
    msmol_fragments = msmol.get_fragments_from_single_cut(include_ring_cuts=True)
    b = np.nonzero(fragments_map)
    print(b)
    input()