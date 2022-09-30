import os
import shutil
import time
import requests
from flask_apscheduler import APScheduler
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import requests
from flask import Flask, render_template, request, redirect, flash, send_from_directory, Markup
from werkzeug.utils import secure_filename
from utils.fertilizer import fertilizer_dict
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data import disease_map, details_map

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


# Load model from downloaded model file
model = load_model('model.h5')

# Create folder to save images temporarily
if not os.path.exists('./static/test'):
        os.makedirs('./static/test')

def predict(test_dir):
    test_img = [f for f in os.listdir(os.path.join(test_dir)) if not f.startswith(".")]
    test_df = pd.DataFrame({'Image': test_img})
    
    test_gen = ImageDataGenerator(rescale=1./255)

    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        test_dir, 
        x_col = 'Image',
        y_col = None,
        class_mode = None,
        target_size = (256, 256),
        batch_size = 20,
        shuffle = False
    )
    predict = model.predict(test_generator, steps = np.ceil(test_generator.samples/20))
    test_df['Label'] = np.argmax(predict, axis = -1) # axis = -1 --> To compute the max element index within list of lists
    test_df['Label'] = test_df['Label'].replace(disease_map)

    prediction_dict = {}
    for value in test_df.to_dict('index').values():
        image_name = value['Image']
        image_prediction = value['Label']
        prediction_dict[image_name] = {}
        prediction_dict[image_name]['prediction'] = image_prediction
        prediction_dict[image_name]['description'] = details_map[image_prediction][0]
        prediction_dict[image_name]['symptoms'] = details_map[image_prediction][1]
        prediction_dict[image_name]['source'] = details_map[image_prediction][2]
    return prediction_dict


# Create an app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # maximum upload size is 50 MB
app.secret_key = "agentcrop"
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}
folder_num = 0
folders_list = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# initialize scheduler
scheduler = APScheduler()
scheduler.api_enabled = True
scheduler.init_app(app)

# Adding Interval Job to delete folder
@scheduler.task('interval', id='clean', seconds=1800, misfire_grace_time=900)
def clean():
    global folders_list
    try:
        for folder in folders_list:
            if (time.time() - os.stat(folder).st_ctime) / 3600 > 1:
                shutil.rmtree(folder)
                folders_list.remove(folder)
                print("\n***************Removed Folder '{}'***************\n".format(folder))
    except:
        flash("Something Went Wrong! couldn't delete data!")

scheduler.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/404.html")
def error_html():
    return render_template("404.html")

@app.route('/', methods=['GET','POST'])
def get_disease():
    global folder_num
    global folders_list
    if request.method == 'POST':
        if folder_num >= 1000000:
            folder_num = 0
        # check if the post request has the file part
        if 'hiddenfiles' not in request.files:
            flash('No files part!')
            return redirect(request.url)
        # Create a new folder for every new file uploaded,
        # so that concurrency can be maintained
        files = request.files.getlist('hiddenfiles')
        app.config['UPLOAD_FOLDER'] = "./static/test"
        app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            folders_list.append(app.config['UPLOAD_FOLDER'])
            folder_num += 1
        for file in files:
            if file.filename == '':
                flash('No Files are Selected!')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                flash("Invalid file type! Only PNG, JPEG/JPG files are supported.")
                return redirect('/')

        
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
            diseases = predict(app.config['UPLOAD_FOLDER'])
            return render_template('show_prediction.html', 
            folder = app.config['UPLOAD_FOLDER'], 
            predictions = diseases)
       
    return render_template('index.html')
    

@app.route('/Smart-Plant-icon.png')

def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'Smart-Plant-icon.png')

#API requests are handled here
@app.route('/api/predict', methods=['POST', 'GET'])
def api_predict():
    global folder_num
    global folders_list
    if request.method == "POST":
        if folder_num >= 1000000:
                folder_num = 0
        # check if the post request has the file part
        if 'files' not in request.files:
            return {"Error": "No files part found."}
        # Create a new folder for every new file uploaded,
        # so that concurrency can be maintained
        files = request.files.getlist('files')
        app.config['UPLOAD_FOLDER'] = "./static/test"
        app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            folders_list.append(app.config['UPLOAD_FOLDER'])
            folder_num += 1
        for file in files:
            if file.filename == '':
                return {"Error": "No Files are Selected!"}
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                return {"Error": "Invalid file type! Only PNG, JPEG/JPG files are supported."}

    if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
        diseases = predict(app.config['UPLOAD_FOLDER'])
        return diseases

@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)


def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict_classes(test_image)
        return result
    except:
        return 'x'


@app.route("/predict_pest", methods=['GET', 'POST'])
def predict_pest():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
        if pred == 'x':
            return render_template('unaptfile.html')
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')

if __name__=='_main_':
    app.run(debug=True)
