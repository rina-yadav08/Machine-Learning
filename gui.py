from tkinter import Label, Entry, Button, END, Tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='Enter File path:', fg='blue', font=("Helvetica", 16))
        self.lbl1.place(x=100, y=50)
        self.t1=Entry(bd=5)
        self.t1.place(x=200, y=50)        
        
        self.lbl2=Label(win, text='Enter Row number:', fg='blue', font=("Helvetica", 16))
        self.lbl2.place(x=80, y=100)
        self.t2=Entry(bd=5)
        self.t2.place(x=200, y=100)

        self.btn1 = Button(win, text='Check', fg='red', font=("Helvetica", 16), command=self.predict)
        self.btn1.place(x=200, y=150)
        
        self.lbl3=Label(win, text='Result:')
        self.lbl3.place(x=150, y=200)
        self.t3=Entry()
        self.t3.place(x=200, y=200)
    
    def predict(self):
        self.t3.delete(0, 'end')
        
        # Load the data
        dataframe = pd.read_excel(str(self.t1.get()), header= None)
        data = np.array(dataframe.drop([0,1,2,3,4,5], axis=1))
        
        # Feature Extraction
        x_pred = []
        for i in range(data.shape[0]):
            powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data[i], Fs=1)
            plt.close()
            x_pred.append([np.sum(powerSpectrum),np.argmax(powerSpectrum),np.where(powerSpectrum == np.amax(powerSpectrum))[1][0]])
        
        # Load the Model
        nbc_loaded = load('bayes_classifier_model.joblib')
        
        # Predict using the loaded model
        y_pred = nbc_loaded.predict(x_pred)
        
        if (y_pred[int(self.t2.get())] == 0):
            self.t3.insert(END, str('This is Object 1'))
        else:
            self.t3.insert(END, str('This is Not Object 1'))     

window=Tk()
mywin=MyWindow(window)
window.title('Discrimination of reflected sound signals - QTFR')
window.geometry("400x300+10+10")
window.mainloop()
