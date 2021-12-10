import numpy as np
from modules import Descriptors #, MolFromSmiles
from deepchem.feat.base_classes import MolecularFeaturizer
import logging
from deepchem.utils.typing import RDKitMol


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


##smiles = ["C", "CCC"]
##mol = MolFromSmiles("CCC")
##featurizer = RDKitDescriptors()
##x = featurizer.featurize(mol)
##print(x.shape)
#new_file(x, 'RDKitDescriptors')
