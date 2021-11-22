from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle
from zero_columns import zero_cols_cho, zero_cols_nhal

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
    if request.method == 'POST':
        try:
            smiles = request.form.get('smiles')
            if smiles == 'n':
                return redirect(url_for('no_smiles'))    
            else:    
                Tm = (request.form.get('Tm'))
                pickle_in1 = open('pickle/mol_from_smiles', 'rb')
                converter = pickle.load(pickle_in1)
                mol = converter.molsmiles(smiles)
                pickle_in2 = open('pickle/rdkit_des','rb')
                feat = pickle.load(pickle_in2)
                fp = feat.featurize(mol)
                fp = fp.reshape((208,1))

                if Tm == 'n':
                    model = load('new/mod_2_nhal_no_tm')
                    y_pred = model.predict(fp.transpose())
                    return render_template('results2.html', Tg=y_pred)
                else:
                    X_pred = np.concatenate(([[float(Tm)]], fp.reshape((fp.shape[0], 1))), axis=0)                
                    X_pred = X_pred.transpose()
                    model = load('new/mod_2_nhal')
                    y_pred = model.predict(X_pred)
                    return render_template('results2.html', Tg=y_pred)	
        except:
            return render_template('results2.html', Tg='Schlumpf!')
    else:
        return render_template("home_2.html")

@app.route('/no_smiles', methods=['GET', 'POST'])
def no_smiles():
    if request.method == 'POST':
        try:
            CH3 = int(request.form.get('CH3'))
            CH2 = int(request.form.get('CH2'))
            CH = int(request.form.get('CH'))
            C = int(request.form.get('C'))
            OH = int(request.form.get('OH'))
            COC = int(request.form.get('COC'))
            CO = int(request.form.get('C=O'))
            N = int(request.form.get('N'))
            Hal = int(request.form.get('Hal'))
            DBE = int(request.form.get('DBE'))
            Tm = request.form.get('Tm')
            Ct = CH3 + CH2 + CH + C
            O = OH + COC + CO
            
            if N == 0 and Hal == 0:
                H = 2*(Ct - DBE + 1)
                if Tm == 'n':
                    X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE]                            
                    X_pred.extend([Ct, H, O])
                    X_pred.insert(8, O/Ct)
                    X_pred.insert(9, 12*Ct + H + 16*O)                
                    model = load('new/mod_1_cho_no_tm')
                    y_pred = model.predict(np.array([X_pred]))                
                    return render_template('results2.html', Tg=y_pred)
                else:
                    X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,float(Tm)]                            
                    X_pred.extend([Ct, H, O])
                    X_pred.insert(8, O/Ct)
                    X_pred.insert(9, 12*Ct + H + 16*O)                
                    model = load('new/mod_1_cho')
                    y_pred = model.predict(np.array([X_pred]))                
                    return render_template('results2.html', Tg=y_pred)
            else:
                H = 2*(Ct - DBE + 1) + N - Hal
                if Tm == 'n':        
                    X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,N,Hal]            
                    X_pred.extend([Ct, H, O])
                    X_pred.insert(10, O/Ct)
                    X_pred.insert(11, 12*Ct + H + 16*O)                
                    model = load('new/mod_1_nhal_no_tm')
                    y_pred = model.predict(np.array([X_pred]))        
                    return render_template('results2.html', Tg=y_pred)
                else:
                    X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,N,Hal,float(Tm)]            
                    X_pred.extend([Ct, H, O])
                    X_pred.insert(10, O/Ct)
                    X_pred.insert(11, 12*Ct + H + 16*O)                
                    model = load('new/mod_1_nhal')
                    y_pred = model.predict(np.array([X_pred]))        
                    return render_template('results2.html', Tg=y_pred)
        except:
            return render_template('results2.html', Tg='Schlumpf!')
    else:
        return render_template('no_smiles.html')


if __name__ == '__main__':
        app.run()
