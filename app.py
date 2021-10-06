from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle

#auth = Blueprint('auth', __name__)

app = Flask(__name__)

def load(model_name):
        pickle_in = open('pickle/{}'.format(model_name),'rb')
        model = pickle.load(pickle_in)
        return model

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/cho1', methods=['GET', 'POST'])
def cho1():
    if request.method == 'POST':
        CH3 = int(request.form.get('CH3'))
        CH2 = int(request.form.get('CH2'))
        CH = int(request.form.get('CH'))
        C = int(request.form.get('C'))
        OH = int(request.form.get('OH'))
        COC = int(request.form.get('COC'))
        CO = int(request.form.get('C=O'))
        DBE = int(request.form.get('DBE'))
        Tm = float(request.form.get('Tm'))

        X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,Tm]
        Ct = CH3 + CH2 + CH + C
        H = 2*(Ct - DBE + 1)
        O = OH + COC + CO
        X_pred.extend([Ct, H, O])
        X_pred.insert(8, O/Ct)
        X_pred.insert(9, 12*Ct + H + 16*O)
        X_pred = np.array([X_pred])

        print('X =', X_pred)
        model = load('Modell_1/extra_trees_CHO_new_best')
        y_pred = model.predict(X_pred)
        print('Tg =', y_pred)
        return render_template('results.html', Tg=y_pred)
    else:
        return render_template("cho1.html")

@app.route('/results')
def results():
        return render_template('results.html')


@app.route('/logout')
#@login_required
def logout():
    return redirect(url_for('/'))


@app.route('/cho2', methods=['GET', 'POST'])
def cho2():
    if request.method == 'POST':
        smiles = request.form.get('smiles')
        M = float(request.form.get('M'))
        Tm = float(request.form.get('Tm'))
        print('Lump')
        print(smiles, type(smiles))
        print(M, type(M))
        print(Tm, type(Tm))
    else:
        print('Schlumpf')

    return render_template("cho2.html")

@app.route('/N+hal_1', methods=['GET', 'POST'])
def N_hal_1():
    if request.method == 'POST':
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
        Tm = float(request.form.get('Tm'))

        X_pred = [CH3,CH2,CH,C,OH,COC,CO,DBE,N,Hal,Tm]
        Ct = CH3 + CH2 + CH + C
        H = 2*(Ct - DBE + 1)
        O = OH + COC + CO
        X_pred.extend([Ct, H, O])
        X_pred.insert(10, O/Ct)
        X_pred.insert(11, 12*Ct + H + 16*O)
        X_pred = np.array([X_pred])

        print('X =', X_pred)
        model = load('Modell_1/extra_trees_nhal_new_best')
        y_pred = model.predict(X_pred)
        print('Tg =', y_pred)
        return render_template('results.html', Tg=y_pred)
    else:
        return render_template('N+hal_1.html')

@app.route('/N+hal_2', methods=['GET', 'POST'])
def N_hal_2():
        return render_template('N+hal_2.html')

if __name__ == '__main__':
        app.run()
