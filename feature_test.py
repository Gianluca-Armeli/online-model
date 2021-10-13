import numpy as np
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

def new_file(df, file_name):
    nf = open('{}.txt'.format(file_name),'w+')
    for i in df:
        nf.write(' '.join(map(str,i)) + '\n')
    nf.close()
'''
smiles = ["C", "CCC"]
featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
f = featurizer.featurize(smiles)

mols = ["C", "CCC"]
featurizer = dc.feat.WeaveFeaturizer()
X = featurizer.featurize(mols)
print(X)
'''
mol = Chem.MolFromSmiles("CCC")
atom = mol.GetAtoms()[2]
#print(dc.feat.graph_features.get_feature_list(atom))
# Using ConvMolFeaturizer to create featurized fragments derived from molecules of interest.
# This is used only in the context of performing interpretation of models using atomic
# contributions (atom-based model interpretation)
'''
smiles = ["C", "CCC"]
featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
f = featurizer.featurize(smiles)
print(len(f)) # contains 2 lists with  featurized fragments from 2 mols
'''
class myMolSmiles():
	def __init__(self):
		pass

	def molsmiles(self, smiles):
		from rdkit import Chem
		mol = Chem.MolFromSmiles(smiles)
		return mol


class RDKitDescriptors(MolecularFeaturizer):

    def __init__(self, use_fragment=True, ipc_avg=True):
	    self.use_fragment = use_fragment
	    self.ipc_avg = ipc_avg
	    self.descriptors = []
	    self.descList = []
        
    def _featurize(self, mol: RDKitMol) -> np.ndarray:
	    """
	    Calculate RDKit descriptors.
	    Parameters
	    ----------
	    mol: rdkit.Chem.rdchem.Mol
	      RDKit Mol object
	    Returns
	    -------
	    np.ndarray
	      1D array of RDKit descriptors for `mol`.
	      The length is `len(self.descriptors)`.
	    """
	    # initialize
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

	    #mol = Chem.MolFromSmiles(smiles)

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

class OneHotFeaturizer(Featurizer):

    def __init__(self, charset: List[str] = ZINC_CHARSET, max_length: int = 100):
        if len(charset) != len(set(charset)):
            raise ValueError("All values in charset must be unique.")
        self.charset = charset
        self.max_length = max_length

    def featurize(self, datapoints: Iterable[Any], log_every_n: int = 1000) -> np.ndarray: 
        datapoints = list(datapoints)
        if (len(datapoints) < 1):
            return np.array([]) # Featurize data using featurize() in grandparent class
        return Featurizer.featurize(self, datapoints, log_every_n)

    def _featurize(self, datapoint: Any): # Featurize str data
        if (type(datapoint) == str):
            return self._featurize_string(datapoint)
        # Featurize mol data
        else:
            return self._featurize_mol(datapoint)

    def _featurize_string(self, string: str) -> np.ndarray:
    # validation
	    if (len(string) > self.max_length):
	        logger.info("The length of {} is longer than `max_length`. So we return an empty array.")
	        return np.array([])

	    string = self.pad_string(string)  # Padding
	    return np.array([
	        one_hot_encode(val, self.charset, include_unknown_set=True)
	        for val in string])

    def _featurize_mol(self, mol: RDKitMol) -> np.ndarray:
	    try:
	        from rdkit import Chem
	    except ModuleNotFoundError:
	        raise ImportError("This class requires RDKit to be installed.")
	    smiles = Chem.MolToSmiles(mol)  # Convert mol to SMILES string.
	    return self._featurize_string(smiles)  # Use string featurization.
    def pad_smile(self, smiles: str) -> str:
        return self.pad_string(smiles)

    def pad_string(self, string: str) -> str:
        return string.ljust(self.max_length)
    
    def untransform(self, one_hot_vectors: np.ndarray) -> str:
	    string = ""
	    for one_hot in one_hot_vectors:
	        try:
	            idx = np.argmax(one_hot)
	            string += self.charset[idx]
	        except IndexError:
	            string += ""
	    return string

# smiles = ["C", "CCC"]
# featurizer = RDKitDescriptors()
# x = featurizer.featurize(mol)
# #print(x.shape)
# new_file(x, 'RDKitDescriptors')