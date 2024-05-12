# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Backend Side
# ---------------------------
# ---------------------------
# # # # # # # # # # # # # # # 

from io import BytesIO
import os
from PIL import Image
import numpy as np
from multiprocessing import Process, freeze_support
import threading
from flask import Flask, request, jsonify, render_template
from dltgui.dlgui import dl_gui
import tensorflow as tf
import binascii



# Set Flask
app = Flask(__name__, static_folder="static/")



@app.route('/')
def home():
    return render_template('index.html', title ="Version 1.0.2")

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/terminal-object-detection',methods = ['POST'])
def terminal_object_detection():
    if request.method == 'POST':
      result = request.form
      dataset = result['dataset']
      type_of_label = result['type_of_label']
      split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      return render_template("terminal-object-detection.html",result = result)


@app.route('/terminal',methods = ['POST'])
def terminal():

   if request.method == 'POST':
      '''Read the values from HTML file and set the values for training.''' 
      result = request.form
      dataset = result['dataset']
      split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      cpu_gpu = result['CPU/GPU']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      activation_function = result['activation_function']
      flip = result['flip'] 
      rotation = result['rotation'] 
      zoom = result['zoom']
      fine_tuning = result['fine_tuning']
      fine_tune_epochs = result['fine_tune_epochs']
      img_height = result['img_height']
      img_width = result['img_height']
      gui = dl_gui(dataset=dataset, 
                   project_name = str(project_name), 
                   split_dataset = float(split_dataset), 
                   pre_trained_model = pre_trained_model,
                   cpu_gpu = cpu_gpu,
                   number_of_classes = int(number_of_classes), 
                   batch_size = int(batch_size), 
                   epoch = int(epoch),
                   activation_function = activation_function,
                   fine_tune_epochs = int(fine_tune_epochs),
                   img_height=int(img_height),
                   img_width=int(img_width))

      if(flip == "True" or rotation == "True" or zoom == "True"):
          samples = result['samples']
          gui.load_dataset(imgaugmentation = True, flip = flip, rotation = rotation, zoom = zoom, samples = int(samples))
      else:
          gui.load_dataset()
        
      thread_gui = threading.Thread(target= gui.train, args=(fine_tuning,)).start()
      return render_template("terminal.html",result = result, thread_gui = thread_gui)

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/convert', methods = ['GET', 'POST'])
def convert2TFlite():
    def convert_to_c_array(bytes) -> str:
        hexstr = binascii.hexlify(bytes).decode("UTF-8")
        hexstr = hexstr.upper()
        array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
        array = [array[i:i+10] for i in range(0, len(array), 10)]
        return ",\n  ".join([", ".join(e) for e in array])

    if request.method == 'GET':
        return render_template('convert.html')
    elif request.method == 'POST':
        result = request.form
        model_name= result['model']
        if model_name and model_name.endswith('h5'):
            model01 = tf.keras.models.load_model('models/' + model_name)
            converter = tf.lite.TFLiteConverter.from_keras_model(model01)
            tflite_model = converter.convert()
            model_lite_name = 'models/'+ os.path.splitext(model_name)[0]+ '.tflite'
            open(model_lite_name, "wb").write(tflite_model)
            # convert tflite format to cc file
            tflite_binary = open(model_lite_name, 'rb').read()
            ascii_bytes = convert_to_c_array(tflite_binary)
            c_file = "const unsigned char tf_model[] = {\n  " + ascii_bytes + "\n};\nunsigned int tf_model_len = " + str(len(tflite_binary)) + ";"
            #print(c_file)
            model_h_name = 'models/'+ os.path.splitext(model_name)[0]+ '.h'
            open(model_h_name, "w").write(c_file)
            os.remove(model_lite_name)
            model_name = ''
        else:
            model_lite_name = ''
        return render_template('convert.html', model_h_name = model_h_name)

@app.route('/result', methods = ['POST'])
def result():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    if request.method == 'POST':
        result = request.form
        dataset = result['dataset']
        model_dir= result['model']
        img = result['img']
        gui = dl_gui(project_name = "Test", dataset=dataset)
        predicted_class, max_pred, show_heatmap, img_name = gui.predict(img, model_dir)
        return render_template('result.html', result = result, img = img, max_pred = max_pred, predicted_class = predicted_class, show_heatmap = show_heatmap, img_name = img_name,  mimetype="text/event-stream")

@app.route('/test', methods = ['POST'])
def test():
     if request.method == 'POST':
      '''Read the values from HTML file and set the values for training.''' 
      result = request.form
      dataset = result['dataset']
      split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      cpu_gpu = result['CPU/GPU']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      activation_function = result['activation_function'] 
      flip = result['flip'] 
      rotation = result['rotation'] 
      zoom = result['zoom'] 
      print("Flip:   ", flip, rotation,zoom)
      return "Testing page - look at the conda terminal for values.."

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        # Read the image file and convert it to PIL format
        img = Image.open(BytesIO(file.read()))

        # if not os.path.exists('uploads'):
        #     os.makedirs('uploads')
        # Do something with the PIL image (e.g., save it, process it, etc.)
        img.save('static/'+ file.filename)  # Save the image

        return 'Image uploaded successfully'

    return 'Upload failed'


if __name__ == "__main__":
    freeze_support()
    app.run(debug=True)
   
   