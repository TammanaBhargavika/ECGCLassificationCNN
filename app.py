import pandas as pd
import tensorflow as tf
from flask import Flask,render_template,request
import os

app=Flask(__name__)

new_model=tf.keras.models.load_model('./model/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():        
    if request.method=='POST':
        f=request.files['file']
        f.save('./upload/input_file.csv')

    x=pd.read_csv('./upload/input_file.csv',header=None)

    x_test=x.iloc[:,:187].values
    ref=x.iloc[:,:187]
    x_test=x_test.reshape(len(x_test),x_test.shape[1],1)

    y=new_model.predict(x_test)
    yp=y.argmax(axis=1)
    ref[187]=yp
    ref.to_csv('./upload/output.csv',header=False,index=False)

    restring=[]
    
    for i in yp:
        if i==0:
            restring.append('Normal beat')
        elif i==1:
            restring.append('Supraventricular ectopic beats')
        elif i==2:
            restring.append('Ventricular ectopic beats')
        elif i==3:
            restring.append('Fusion beats')
        elif i==4:
            restring.append('Unknown beats')

    restring=",".join(restring)

    os.remove('./upload/output.csv')

    return render_template('result.html',name=restring)

if __name__== "__main__":
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.run(debug=True)
