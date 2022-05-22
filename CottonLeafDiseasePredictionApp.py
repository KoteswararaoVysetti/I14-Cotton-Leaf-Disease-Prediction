
from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_cottonLeaf_disease(cotton_leaf):
  test_image = load_img(cotton_leaf, target_size = (150, 150)) 
  
  test_image = img_to_array(test_image)/255 
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) 
  print('@@ Raw result = ', result)
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return "CottonLeaf - Ascochyta Disease", 'CottonLeaf-Ascochyta_Blight.html'
       
  elif pred==1:
      return "CottonLeaf - Bacterial Blight Disease", 'CottonLeaf-Bacterial_Blight.html'
        
  elif pred==2:
      return "CottonLeaf - Healthy and Fresh", 'CottonLeaf-Healthy.html'
        
  elif pred==3:
      return "CottonLeaf - Target_Spot Disease", 'CottonLeaf-Target_Spot.html'
       
  elif pred==4:
      return "CottonLeaf - Cercospora Disease", 'CottonLeaf-Cercospora.html'
        

    

#flask instance
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] 
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/InputImages', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cottonLeaf_disease(cotton_leaf=file_path)
        print(pred,output_page);
        return render_template(output_page, pred_output = pred, user_image = file_path)
    

if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
    
    
