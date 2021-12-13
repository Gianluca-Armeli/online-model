from flask import Flask, render_template, request, flash, redirect, url_for
from rdkit.Chem import MolFromSmiles
#from feature_test import RDKitDescriptors
import numpy as np
import pickle
import deepchem as dc
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.complex_featurizers import ComplexNeighborListFragmentAtomicCoordinates
from deepchem.feat.mol_graphs import ConvMol, WeaveMol
from deepchem.data import DiskDataset
import logging
from typing import Optional, List, Union, Iterable
from deepchem.utils.typing import RDKitMol, RDKitAtom
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
import logging
from typing import List
from deepchem.utils.typing import RDKitMol
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.feat.base_classes import Featurizer
from typing import Any, Iterable

app = Flask(__name__)

def load(model_name):
        pickle_in = open('pickle/{}'.format(model_name),'rb')
        model = pickle.load(pickle_in)
        return model
        

@app.route('/results')
def results():
    return render_template('results2.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    # if request.method == 'POST':
    # #try:
    #     smiles = request.form.get('smiles')
    #     if smiles == 'n':
    #         return redirect(url_for('no_smiles'))    
    #     else:    
    #         Tm = (request.form.get('Tm'))
    #         mol = MolFromSmiles(smiles)
    #         featurizer = RDKitDescriptors()
    #         fp = featurizer.featurize(mol)
    #         fp = fp.reshape((208,1))

    #         if Tm == 'n':
    #             model = load('new/mod_2_nhal_no_tm')
    #             y_pred = model.predict(fp.transpose())
    #             return render_template('results2.html', Tg=str(round(y_pred[0], 1))+' K')
    #         else:
    #             X_pred = np.concatenate(([[float(Tm)]], fp.reshape((fp.shape[0], 1))), axis=0)                
    #             X_pred = X_pred.transpose()
    #             model = load('new/mod_2_nhal')
    #             y_pred = model.predict(X_pred)

    #             return render_template('results2.html', Tg=str(round(y_pred[0], 1))+' K')	
    # #except:
    #  #   return render_template('results2.html', Tg='Schlumpf!')
    # else:
    return render_template("home_2.html")

# @app.route('/no_smiles', methods=['GET', 'POST'])
# def no_smiles():
#     if request.method == 'POST':
#         #try:
#         CH3 = int(request.form.get('CH3'))
#         CH2 = int(request.form.get('CH2'))
#         CH = int(request.form.get('CH'))
#         C = int(request.form.get('C'))
#         OH = int(request.form.get('OH'))
#         COC = int(request.form.get('COC'))
#         CO = int(request.form.get('C=O'))
#         N = int(request.form.get('N'))
#         Hal = int(request.form.get('Hal'))
#         DBE = int(request.form.get('DBE'))
#         Tm = request.form.get('Tm')
#         Ct = CH3 + CH2 + CH + C
#         O = OH + COC + CO
        
#         if N == 0 and Hal == 0:
#             H = 2*(Ct - DBE + 1)
#             if Tm == 'n':
#                 X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE]                            
#                 X_pred.extend([Ct, H, O])
#                 X_pred.insert(8, O/Ct)
#                 X_pred.insert(9, 12*Ct + H + 16*O)                
#                 model = load('new/mod_1_cho_no_tm')
#                 y_pred = model.predict(np.array([X_pred]))                
#                 return render_template('results2.html', Tg=str(round(y_pred[0], 1))+' K')
#             else:
#                 X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,float(Tm)]                            
#                 X_pred.extend([Ct, H, O])
#                 X_pred.insert(8, O/Ct)
#                 X_pred.insert(9, 12*Ct + H + 16*O)                
#                 model = load('new/mod_1_cho')
#                 y_pred = model.predict(np.array([X_pred]))                
#                 return render_template('results2.html', Tg=str(round(y_pred[0], 1))+' K')
#         else:
#             H = 2*(Ct - DBE + 1) + N - Hal
#             if Tm == 'n':        
#                 X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,N,Hal]            
#                 X_pred.extend([Ct, H, O])
#                 X_pred.insert(10, O/Ct)
#                 X_pred.insert(11, 12*Ct + H + 16*O)                
#                 model = load('new/mod_1_nhal_no_tm')
#                 y_pred = model.predict(np.array([X_pred]))        
#                 return render_template('results2.html', Tg=str(round(y_pred[0], 1))+' K')
#             else:
#                 X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,N,Hal,float(Tm)]            
#                 X_pred.extend([Ct, H, O])
#                 X_pred.insert(10, O/Ct)
#                 X_pred.insert(11, 12*Ct + H + 16*O)                
#                 model = load('new/mod_1_nhal')
#                 y_pred = model.predict(np.array([X_pred]))        
#                 return render_template('results2.html', Tg=str(round(y_pred[0], 1))+' K')
#         #except:
#             #return render_template('results2.html', Tg='Schlumpf!')
#     else:
#         return render_template('no_smiles.html')


if __name__ == '__main__':
    app.run()
