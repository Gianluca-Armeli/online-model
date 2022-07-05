import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, Descriptors
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
import logging

class RDKitDescriptors(MolecularFeaturizer):

    def __init__(self, use_fragment=True, ipc_avg=True):
        self.use_fragment = use_fragment
        self.ipc_avg = ipc_avg
        self.descriptors = []
        self.descList = []
        
    def _featurize(self, mol: RDKitMol) -> np.ndarray:
        if len(self.descList) == 0:
            try:
                from rdkit.Chem import Descriptors
                for descriptor, function in Descriptors.descList:
                    if self.use_fragment is False and descriptor.startswith('fr_'):
                        continue
                    self.descriptors.append(descriptor)
                    self.descList.append((descriptor, function))
            except ModuleNotFoundError:
                raise ImportError("This class requires RDKit to be installed.")

        # check initialization
        assert len(self.descriptors) == len(self.descList)

        features = []
        for desc_name, function in self.descList:
            if desc_name == 'Ipc' and self.ipc_avg:
                feature = function(mol, avg=True)
            else:
                feature = function(mol)
            features.append(feature)
        return np.asarray(features)


logger = logging.getLogger(__name__)
ZINC_CHARSET = ['#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=',
    '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 'p', 's', 'r']


def make_formula(formula_list):
    sf = ''
    for name, number in formula_list:
        if number != 0:
            sf += name + str(number)
    return sf

def FormFromMol(mol):
    mol = Chem.AddHs(mol)
    sym = [atom.GetSymbol() for atom in mol.GetAtoms()]
    C = sym.count('C')
    H = sym.count('H')
    O = sym.count('O')
    N = sym.count('N')
    F = sym.count('F')
    Br = sym.count('Br')
    Cl = sym.count('Cl')
    I = sym.count('I')
    formula_list = [['C', C], ['H', H], ['O', O], ['N', N],
            ['F', F], ['Cl', Cl], ['Br', Br], ['I', I]]
    formula = make_formula(formula_list)
    return formula

#names = ['smiles', 'Tg']
names = ['smiles', 'Tm', 'Tg']
for i in range(208):
    names.append(str('c' + str(i)))

names2 = ['ch3', 'ch2', 'ch', 'c', 'oh', 'coc', 'c=o', 'n', 'hal', 'dba', 'oc', 'm', 'tm', 'tg']
names3 = ['ch3', 'ch2', 'ch', 'c', 'oh', 'coc', 'c=o', 'dba', 'n',
          'hal', 'oc', 'm', 'tm', 'tg']

test='C1=CC=CC=C1'
test2 = str(np.array([2, 6, 1, 0, 1, 0, 0, 0, 0.11111, 144]))
test3 = str(np.array([1, 0, 9, 6, 0, 0, 3.5, 12, 2, 0, 0.21875, 266, 395]))

df = pd.read_csv('train_nhal_mod_2.txt', delim_whitespace=True, names=names)
df2 = pd.read_csv('train_cho_mod_1.txt', delim_whitespace=True, names=names2)
df3 = pd.read_csv('train_nhal_mod_1.txt', delim_whitespace=True, names=names3)

df2_X = np.array(df2.drop(['tg'], axis=1))
df2_X_notm = np.array(df2.drop(['tm', 'tg'], axis=1))

df3_X = np.array(df3.drop(['tg'], axis=1))
df3_X_notm = np.array(df3.drop(['tm', 'tg'], axis=1))

dic = {}
for i in range(len(df)):
    dic.update({df['smiles'][i]:float(df['Tg'][i])})

dic_ns_cho = {}
for i in range(len(df2)):
    dic_ns_cho.update({str(df2_X[i]):float(df2['tg'][i])})
    dic_ns_cho.update({str(df2_X_notm[i]):float(df2['tg'][i])})

dic_ns_nhal = {}
for i in range(len(df3)):
    dic_ns_nhal.update({str(df3_X[i]):float(df3['tg'][i])})
    dic_ns_nhal.update({str(df3_X_notm[i]):float(df3['tg'][i])})


featurizer = RDKitDescriptors()

def seek_duplicates(mode, test):
    sack = False
    if mode == 'SMILES mode with Tm' or mode == 'SMILES mode, no Tm':
        mol_test = MolFromSmiles(test)
        fp_test = featurizer.featurize(mol_test)
        for string in dic:
            mol_dic = MolFromSmiles(string)
            fp_dic = featurizer.featurize(mol_dic)
            comp = fp_dic == fp_test
            equal_arrays = comp.all()
            if equal_arrays:
                sack = True

    elif mode == 'no SMILES, CH/CHO compound with Tm' or mode == 'no SMILES, CH/CHO compound, no Tm':
        for j in range(len(df2)):
            #comp = df2_X[j] == test
            #equal_arrays = comp.all()
            
            #if equal_arrays:
            if test == str(df2_X[j]) or test == str(df2_X_notm[j]):
                sack = True
    else:
        for i in range(len(df3)):
            #comp = df3_X[i] == test
            #equal_arrays = comp.all()
            if test == str(df3_X[i]) or test == str(df3_X_notm[i]):
                sack = True
    return sack

#print(seek_duplicates('SMILES mode with Tm', test))
#print(seek_duplicates('no pound with Tm', test3))
