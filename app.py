from flask import Flask, render_template, request, flash, redirect, url_for
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
import pickle
import warnings
import numpy as np
from methods import *
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'das wiesel läuft um mitternacht'

def load(model_name):
        pickle_in = open('pickle/{}'.format(model_name),'rb')
        model = pickle.load(pickle_in)
        return model
  
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/smiles', methods = ['GET', 'POST'])
def smiles():
    if request.method == 'POST':
        #try:
        smiles = request.form.get('smiles')
        if smiles == '':
            flash('SMILES input field can\'t be empty!', category='error')
            return redirect(url_for('smiles'))    
        elif smiles.isnumeric() or ('C' not in smiles and smiles != 'O'):
            flash('Invalid SMILES string!', category='error')
            return redirect(url_for('smiles'))    
        else:
            smiles = smiles.strip()
            mode = 'SMILES mode, no Tm'
            if seek_duplicates(mode, smiles):
                flash('This compound is part of the training set. Tg literature value: '+str(dic[smiles])+' K', category='note')
            Tm = request.form.get('Tm')
            mol = MolFromSmiles(smiles)
            featurizer = RDKitDescriptors()
            fp = featurizer.featurize(mol)
            fp = fp.reshape((208,1))
            M = round(MolWt(mol), 1)

            if Tm == '':
                model = load('mod_2_nhal_no_tm')
                y_pred = model.predict(fp.transpose())
                return render_template('results.html', mode=mode,
                                        sum_formula=FormFromMol(mol), M=str(M),  Tg=str(round(y_pred[0], 1)), mae='15.1')
            else:
                if not Tm.lstrip('-').isdigit():
                    flash('Tm must be a number!', category='error')
                    return render_template("smiles.html")
                elif float(Tm) <= 0:
                    flash('Tm must be positive!', category='error')
                    return render_template("smiles.html")
                elif float(Tm) > 666:
                    flash('Tm value is too large!', category='error')
                    return render_template("smiles.html")
                else:
                    X_pred = np.concatenate(([[float(Tm)]], fp.reshape((fp.shape[0], 1))), axis=0)                
                    X_pred = X_pred.transpose()
                    model = load('mod_2_nhal')
                    y_pred = model.predict(X_pred)

                    return render_template('results.html', mode='SMILES mode with Tm',
                                           sum_formula=FormFromMol(mol), M=str(M), Tg=str(round(y_pred[0], 1)), mae='11.7')	
        #except:
        #    return redirect(url_for('error'))

    else:
        return render_template("smiles.html")

@app.route('/no_smiles', methods=['GET', 'POST'])
def no_smiles():
    if request.method == 'POST':
        #try:
        Hal = 0
        hal_list = []
        hal = []
        hal_list.append(request.form.get('F'))
        hal_list.append(request.form.get('Cl'))
        hal_list.append(request.form.get('Br'))
        hal_list.append(request.form.get('I'))
        for h in hal_list:
            if h == '':
                hal.append(0)
            else:
                hal.append(int(h))

        Hal = sum(hal)
        feat = []
        feat.append(request.form.get('CH3')) 
        feat.append(request.form.get('CH2'))
        feat.append(request.form.get('CH'))
        feat.append(request.form.get('C'))
        feat.append(request.form.get('OH'))
        feat.append(request.form.get('COC'))
        feat.append(request.form.get('C=O'))
        feat.append(request.form.get('N'))
        feat.append(request.form.get('DBE'))
        Tm = request.form.get('Tm')
        if Tm == '':
            Tm = 1000
        
        X_pred = []
        for f in feat:
            if f == '':
                X_pred.append(0)
            else:
                X_pred.append(int(f))

        X_pred.insert(8, Hal)        
        Ct = X_pred[0] + X_pred[1] + X_pred[2] + X_pred[3]
        O = X_pred[4] + X_pred[5] + X_pred[6]
        
        if X_pred[7] == 0 and X_pred[8] == 0:
            H = 2*(Ct - X_pred[9] + 1)

            formula_list = [['C', Ct], ['H', H], ['O', O]]
            formula = make_formula(formula_list)
            M = 12*Ct + H + 16*O
            X_pred.extend([X_pred[7], X_pred[8], Ct, H, O])
            del X_pred[7:9]
            X_pred.insert(10, float(Tm))
            X_pred.insert(10, round(O/Ct, 5))
            X_pred.insert(11, M) 

            if request.form.get('Tm') == '':
                mode = 'no SMILES, CH/CHO compound, no Tm'
                X_pred.remove(float(Tm))                                 
                if seek_duplicates(mode, str(np.array(X_pred[:-3]))):
                    flash('Input vector is part of the training set. Tg literature value: '
                        +str(dic_ns_cho[str(np.array(X_pred[:-3]))])+' K', category='note')  
                del X_pred[8:10]
                model = load('mod_1_cho_no_tm')
                y_pred = model.predict(np.array([X_pred]))                
                return render_template('results.html', mode=mode,
                                       sum_formula=formula, M=str(M), Tg=str(round(y_pred[0], 1)), mae='17.4')
            else:
                mode = 'no SMILES, CH/CHO compound with Tm'
                if float(Tm) < 0:
                    flash('Tm must be positive!', category='error')
                if seek_duplicates(mode, str(np.array(X_pred[:-4]))):
                    flash('Input vector is part of the training set. Tg literature value: '
                        +str(dic_ns_cho[str(np.array(X_pred[:-4]))])+' K', category='note') 
                del X_pred[8:10]              
                model = load('mod_1_cho')
                y_pred = model.predict(np.array([X_pred]))                
                return render_template('results.html', mode=mode,
                                       sum_formula=formula, M=str(M), Tg=str(round(y_pred[0], 1)), mae='13.3')
        else:
            H = 2*(Ct - X_pred[9] + 1) + X_pred[7] - X_pred[8]
            formula_list = [['C', Ct], ['H', H], ['O', O], ['N', X_pred[7]],
            ['F', hal[0]], ['Cl', hal[1]], ['Br', hal[2]], ['I', hal[3]]]
            formula = make_formula(formula_list)
            X_pred.extend([X_pred[7], X_pred[8], Ct, H, O])
            M = 12*Ct + H + 16*O + X_pred[7]*14 + hal[0]*19 + hal[1]*35.4 + hal[2]*79.9 + hal[3]*126.9 
            del X_pred[7:9]
            X_pred.insert(10, float(Tm))
            X_pred.insert(10, round(O/Ct,5))
            X_pred.insert(11, M)  

            if request.form.get('Tm') == '': 
                mode = 'no SMILES, N/Hal compound, no Tm'       
                X_pred.remove(float(Tm))
                if seek_duplicates(mode, str(np.array(X_pred[:-3]))):
                    flash('Input vector is part of the training set. Tg literature value: '
                        +str(dic_ns_nhal[str(np.array(X_pred[:-3]))])+' K', category='note')                
                model = load('mod_1.2_nhal_no_tm')
                y_pred = model.predict(np.array([X_pred]))        
                return render_template('results.html', mode=mode, 
                    sum_formula=formula, M=str(M), Tg=str(round(y_pred[0], 1)), mae='20.4')
            else:
                mode = 'no SMILES, N/Hal compound with Tm'
                if float(Tm) < 0:
                    flash('Tm must be positive!', category='error')
                if seek_duplicates(mode, str(np.array(X_pred[:-4]))):
                    flash('Input vector is part of the training set. Tg literature value: '
                        +str(dic_ns_nhal[str(np.array(X_pred[:-4]))])+' K', category='note')              
                model = load('mod_1.2_nhal')
                y_pred = model.predict(np.array([X_pred]))        
                return render_template('results.html', mode=mode,
                    sum_formula=formula, M=str(M), Tg=str(round(y_pred[0], 1)), mae='13.0')
       # except:
        #    return redirect(url_for('error'))
    else:
        return render_template('no_smiles.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
