# **SPHERE-MS**

### SPHEREMS: _**S**econdary-loss **P**owered **H**ybrid **E**dge **R**emoval **E**ngine for predicting **M**ass **S**pectra_

### Usage
1. Setup dependencies <br> 
```bash
conda env create -f spherems.yaml
conda activate spherems
```

2. Using SPHERE-MS
We have created a demo folder which includes a csv file `demo_in.csv` containing molecules to be estimated. 
```bash
python run_model.py --input_csv demo/demo_input.csv --out_msp demo/demo_out.msp --ckpt data/trained_SPHEREMS/SPHEREMS_trained.ckpt --hyper data/trained_SPHEREMS/hparams.yaml
```
3. Parameters
   - --input_csv, protein structure file; Each line of `example_input.csv` yields an individual spectrum in the output, and should be four commar-delimited fields: a unique identifier for the spectrum, a SMILES string, a precursor type (`[M+H]+` or `[M-H]-`), NCE (normalized collision energy) and instrument type (`HCD`,`Q-TOF`,`IT-FT/ion trap with FTMS`);
   - --out_msp, the output predicted spectra in .msp format;
   - --ckpt, the model checkpoint for trained SPHERE-MS;
   - --hyper, the model hyperparameters for trained SPHERE-MS.

### Training SPHERE-MS
For licensing reasons, NIST-20 or files derived from it are not included. After purchasing NIST-20, `hr_msms_nist.MSP` can be extracted with the [Library Conversion Tool](https://chemdata.nist.gov/mass-spc/ms-search/Library_conversion_tool.html).
After obtaining `hr_msms_nist.MSP`, using scripts in `data/` to generate the training/validation set for SPHERE-MS. You may need to modify several lines in `analysis_loss.py` and  `nist2data.py` to adjust the definition of file path.
```bash
cd data
python analysis_loss.py # adjust the file path in the script before running
python nist2data.py # adjust the file path in the script before running
```
Suppose that, the PATH processed NIST-20 dataset is $NIST
Finally, train SPHERE-MS with `train_model.py`:
```bash
python train_model.py --nist_dir $NIST --train_val_test data/NIST/train_val_test.json --save_dir PathToTrainedModel # see source code of train_model.py for default hyperparameters
```