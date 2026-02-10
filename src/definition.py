import rdkit.Chem as Chem
import os, json

ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_TBL = Chem.GetPeriodicTable()

ELECTRON_MASS = 0.00054858 # unit: u
ELEMENTS = ["H","C","N","O","P","S","F","Cl","Br","I"]
ELEMENTS_NOISO = ["H","C","N","O","P","S","F","I"]
ELEMENTS_ISO = ["Cl","Br"]
MONO_MASSES = {E:P_TBL.GetMostCommonIsotopeMass(E) for E in ELEMENTS}
CL37_MASS, BR81_MASS = P_TBL.GetMassForIsotope('Cl',37), P_TBL.GetMassForIsotope('Br',81)
CL37_PER, BR81_PER = 0.2423, 0.4931
ELEMENTS_ONEHOT = ["C","N","O","P","S","F","Cl","Br","I"]
HYBRIDIZATION = ['SP','SP2','SP3','SP3D','SP3D2']
HYBRIDIZATION_ONEHOT = ['SP','SP2','SP3','SP3D','SP3D2','OTHERS']
BOND_ONEHOT = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
INSTRUMENT_TYPES = ['HCD','Q-TOF','IT-FT/ion trap with FTMS']
PRECURSOR_TYPES = ['[M+H]+','[M-H]-']

NIST_MSP_KEYWORDS = ['PRECURSOR_TYPE','COLLISION_ENERGY','INCHIKEY','INSTRUMENT_TYPE']
# MSnLIB 'ADDUCT' will be renamed as 'PRECURSOR_TYPE' as instance property
MSNLIB_MGF_KEYWORDS = ['NAME','SMILES','ADDUCT','FRAGMENTATION_METHOD','SPECTYPE','COLLISION_ENERGY']

# NIST save molecule structures in inchi_key
INCHIKEY_JSON = os.path.join(ROOT_DIR,'data/inchi','nist_inchikey.json')
f = open(INCHIKEY_JSON,'r')
INCHIKEY_TABLE = json.load(f)
f.close()

LOSS_FORMULAS = [
    '',
    'H2O', 'H2','CO','H3N','CO2','H4C2','HCN','O','H2C2O','H2CO','H2CO2','H2C','H4C','H2C2',
    'H4O2','H6C3','H4C2O','H3CN','C2O2','H4C2O2','H4CO','HN','H4','H3CNO','H8C4','H2C2O2',
    'H6C2','HCNO','O2','C','H3C2N','H4C3O','CN','H5C2N','H4CO2','H2CO3','HC2NO','H2C3O',
    'C2','NO','H4C3','H2O2','C2O','H6C2O','H2C2O3','H4C3O2','H6C3O','H3C2NO','H2C3O2','H9C3N',
    'H5CN','H2CN2','H6C2O3','H2C2N2','NO2','H4O','H6C3O2','H6C4O','C2O3','H7C2N','H10C5','H6C4',
    'H4CO3','H3C3N','N2','H7C3N','N','H4C4O','H8C3','H2C3O3','H4C4O2','H8C5','H5C2NO','H10C6O5',
    'H6C2O2','H10C4O','HNO','C2O4','H3CNO2','H5C3NO','H4C4','HC2N','H2C3','H4C3O3','H4C2O3','H8C3O',
    'H8C5O','H3C2NO2','C3O3','H3C3NO','H6C4O2','H4CN2','H5C3N','HC2NO2','H6C3O3','H7C6N','C3O2',
    'H8C4O2','H2C4O2','H8C4O', # top-100
    'HCl','HCOCl','H2CNCl','HC2OCl','CNCl','H2Cl2','HCCl','H3C2Cl','HCO2Cl','HC2Cl','Cl2',
    'HF', 'H2F2', 'HCOF', 'CF2', 'H2CNF', 'HCO2F','HCF3','H2COF2','HCF','H3C2F','H3OF',
    'HBr', 'HCOBr', 'H2CNBr', 'CNBr', 'ClBr', 'H3CBr','Br2','HCBr',
    'O2S', 'S', 'H2S', 'CS', 'OS', 'HCNS','H2CS','H2CO2S',
    'H3O4P', 'HO3P', 'H3O3P',
    'HI', 'I2'
]

