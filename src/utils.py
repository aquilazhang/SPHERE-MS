import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from .MS_chem import Peak, Formula
from .definition import *

def NIST_spectrum_iterator(nist_file):
    with open(nist_file,'r') as f:
        block = ''
        for line in f:
            block += line
            if block.endswith('\n\n'):
                yield block.strip()
                block = ''

def MSnLIB_spectrum_iterator(msnlib_file):
    with open(msnlib_file,'r') as f:
        block = ''
        for line in f:
            block += line
            if block.endswith('\n\n'):
                yield block.strip()
                block = ''
                
def pred2peaks(
    pred_intensity:torch.Tensor|np.ndarray,
    formulas_matrix:torch.Tensor|np.ndarray, 
    ion_charge:float, # [M+H]+ - ELECTRON_MASS
    prob_cutoff: float = 0.001,
    isotopo_expansion = False,
) -> list[Peak]:
    if isinstance(pred_intensity, torch.Tensor):
        pred_intensity = pred_intensity.cpu().numpy()
    if isinstance(formulas_matrix, torch.Tensor):
        formulas_matrix = formulas_matrix.cpu().numpy()
    
    pred_intensity = np.where(pred_intensity > prob_cutoff, pred_intensity, 0.0)
    spec_formula_array, spec_intensity = [], []
    for i,j,k in zip(*np.nonzero(pred_intensity)):
        spec_formula_array.append(formulas_matrix[i,j,k,:])
        spec_intensity.append(pred_intensity[i,j,k].item())
    
    # merge peaks share the same formula
    formula_intensity = defaultdict(float)
    for formula_array, intensity in zip(spec_formula_array, spec_intensity):
        formula = Formula.from_array(formula_array=formula_array)
        formula_intensity[str(formula)] += intensity
    mzs, intensity_out, isotopo_flag, peak_formulas = [], [], [], []
    for formula, intensity in formula_intensity.items():
        formula = Formula(formula)
        if formula == Formula(''):
            pass
        else:
            if isotopo_expansion and formula.have_isotope():
                probs, masses = formula.isotope_mass_distribution()
                masses = masses - (ion_charge * ELECTRON_MASS)
                probs = probs * intensity
                flags = [True for _ in range(len(masses))] 
                flags[0] = False
                intensity_out.extend(probs.tolist())
                mzs.extend(masses.tolist())
                isotopo_flag.extend(flags)
                peak_formulas.extend([formula for _ in range(len(masses))])
            else:
                mzs.append(formula.mass - (ion_charge * ELECTRON_MASS))
                intensity_out.append(intensity)
                isotopo_flag.append(False)
                peak_formulas.append(formula)
    intensity_norm = sum(intensity_out)
    out_peaks = [
        Peak(index=idx,mz=mz,intensity=intensity/intensity_norm,ppm=0.,ion_formula=f,is_isotope=isotopo_f)
        for idx, (mz, intensity,f, isotopo_f) in enumerate(zip(mzs,intensity_out,peak_formulas,isotopo_flag))
    ]
    return out_peaks
        
def plotpeaks(
    peaks: list[Peak],
    ref_peaks: list[Peak] = None,
    mz_limit = 1000.0,
    intensity_limit = 0.5,
):
    fig, ax = plt.subplots()
    norm = sum([peak.intensity for peak in peaks])
    for peak in peaks:
        ax.vlines(x=peak.mz,ymin=0.0,ymax=peak.intensity/norm,colors='blue')
    ax.set_xlim(left=0.0,right=mz_limit)
    ax.set_ylim(bottom=0.,top=intensity_limit)
    if ref_peaks is not None:
        ax.axhline(y=0.0,xmin=0.0,xmax=mz_limit,c='black')
        norm = sum([peak.intensity for peak in ref_peaks])
        for peak in ref_peaks:
            ax.vlines(x=peak.mz,ymax=0.0,ymin=-1*peak.intensity/norm,colors='red')
        ax.set_ylim(bottom=-1*intensity_limit,top=intensity_limit)
    return fig

def peaks2ndarray(peaks:list[Peak]) -> np.ndarray:
    """transofrom peaks to ndarray

    Args:
        peaks (list[Peak]) 

    Returns:
        np.ndarray: in shape [num_peak, 2] (mz, intensity)
    """
    intensity_norm = sum([peak.intensity for peak in peaks])
    mzs = [peak.mz for peak in peaks]
    intensity = [peak.intensity/intensity_norm for peak in peaks]
    return np.vstack([mzs,intensity],dtype=np.float32).transpose()

def mspblock_from_peaks(peaks:list[Peak],**metadata):
    metadata = {k: v for k,v in metadata.items()}
    peaks = sorted(peaks,key=lambda peak:peak.mz)
    intensity_norm = sum([peak.intensity for peak in peaks])
    contents=[]
    for k,v in metadata.items():
        contents.append(f'{k}: {v}\n')
    for peak in peaks:
        line = '{:.4f} {:.3f} {}'.format(peak.mz, peak.intensity/intensity_norm,str(peak.ion_formula))
        contents.append(line+'\n')
    contents.append('\n')
    contents = ''.join(contents)
    return contents
