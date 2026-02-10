from bioservices import UniChem
import os,json,tqdm

inchikey_file = '/home/aquila/data/dataset/NIST/nist_inchikey.txt'
inchikey_json = '/disk1/aquila/work/MS/data/nist_inchikey.json'
with open(inchikey_file,'r') as f:
    inchi_keys = [line.strip() for line in f]
unichem = UniChem()
inchis = {}
for inchi_key in tqdm.tqdm(inchi_keys):
    result = unichem.get_inchi_from_inchikey(inchi_key)
    try:
        inchis[inchi_key] = result[0]['standardinchi']
    except:
        continue

with open(inchikey_json,'w') as f:
    json.dump(inchis,f)