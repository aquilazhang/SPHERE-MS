from src.Model import MSModel
import pandas as pd
from pandarallel import pandarallel
import rdkit.Chem as Chem
from src.MS_chem import MSMol, Formula
from src.encoding import MSMolDataBatch
from src.utils import mspblock_from_peaks
from src.definition import LOSS_FORMULAS
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import argparse, logging, torch, time

logger = logging.getLogger()
loss_formulas = [Formula(L) for L in LOSS_FORMULAS]

def sanitizing_input(item:pd.Series):
    smi,precursor_type,instrument_type,nce = item['SMILES'], item['Precursor_type'],item['Instrument_type'],item['NCE']
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logger.log()
        return None
    if 'Name' in item.keys():
        name = item['Name']
    else:
        name = '_'.join(smi,precursor_type,instrument_type)
    return {'Name':name,'SMILES':smi,'Precursor_type':precursor_type,'Instrument_type':instrument_type,'NCE':nce}

class InferenceMolDataset(TorchDataset):
    def __init__(
        self, 
        inputs:pd.DataFrame, # (Name,SMILES,Precursor_type,NCE,Instrument_type) 
    ):
        super().__init__()
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        d_series = self.inputs.iloc[index,:]
        msmol = MSMol(Chem.MolFromSmiles(d_series['SMILES']),precursor_input=d_series['Precursor_type'])
        covariates = {
            'INSTRUMENT_TYPE':d_series['Instrument_type'],'PRECURSOR_TYPE':d_series['Precursor_type'],'NCE':d_series['NCE']
        }
        return d_series.to_dict(), msmol, covariates
    
def inference_collate_fn(inputs):
    metadata = [i[0] for i in inputs]
    msmols = [i[1] for i in inputs]
    covariates = [i[2] for i in inputs]
    return metadata, MSMolDataBatch.from_input_list(msmols,covariates,loss_formulas)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv',type=str)
    parser.add_argument('--out_msp',type=str)
    parser.add_argument('--ckpt',type=str)
    parser.add_argument('--hyper',type=str)
    parser.add_argument('--batch_size',default=64)
    parser.add_argument('--num_workers',default=20)
    args = parser.parse_args()
    pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=args.num_workers)
    
    # 1. sanitizing csv dataframe
    df = pd.read_csv(args.input_csv,index_col=False)
    df = pd.DataFrame([s for s in df.parallel_apply(sanitizing_input, axis=1) if s is not None])

    # 2. load model and dataset
    msmodel = MSModel.load_from_checkpoint(checkpoint_path=args.ckpt,hparams_file=args.hyper)
    dataset = InferenceMolDataset(df)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,drop_last=False,
                            num_workers=args.num_workers,collate_fn=inference_collate_fn)
    
    # 3. run model
    msmodel.eval()
    with torch.no_grad():
        with open(args.out_msp,'w') as msp_out:
            for metadata, msmol_batch in dataloader:
                msmol_batch = msmol_batch.to(msmodel.device)
                pred_peaks = msmodel.predict_step(msmol_batch)
                for pred_p, meta_d in zip(pred_peaks,metadata):
                    block = mspblock_from_peaks(pred_p,**meta_d)
                    msp_out.write(block)
    end_time = time.time()
    print("Elapsed time: {:.1f}".format((end_time-start_time)/60))