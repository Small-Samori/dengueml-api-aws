from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename

from padelpy import from_smiles
import pandas as pd
import pickle
import json
import os

with open('descriptors.txt') as f:
    a = f.readlines()
cols = [i[:-1] for i in a]
model = pickle.load(open('Random Forest.sav', 'rb'))

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'uploads'

def compute_descriptor(smile_cpd):
    descriptors = from_smiles(smile_cpd)
    descriptors_df = pd.DataFrame(descriptors, index=range(len(smile_cpd)))

    return descriptors_df[cols]

def predict(descriptors):
    class_pred = model.predict(descriptors.values)
    prob_pred = model.predict_proba(descriptors.values)

    return class_pred, prob_pred


@app.route('/', methods=['POST'])
def result_single():
    '''
    if request.method == 'POST':
        cpd = request.get_json()    

        descriptor_df = compute_descriptor([cpd['cpd']])
        prediction, confidence = predict(descriptor_df)

        output_dict_single = {'prediction':str(int(prediction[0])),
                        'confidence':str(max(confidence[0]))
        }
        output_json_object_single = json.dumps(output_dict_single, indent = 4)

        response = app.response_class(
            response=output_json_object_single,
            status=200,
            mimetype='application/json'
        )
        return response
        '''
    return 'Hello World'

@app.route('/multi', methods=['POST'])
def result_multi():
    '''
    if request.method == 'POST':
        uploaded_file = request.files['cpd']

        filename = secure_filename(uploaded_file.filename)

        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            
            filepath = os.path.join(app.config['UPLOAD_PATH'], filename)
            uploaded_file.save(filepath)

            with open(filepath, 'r') as f:
                file_content = f.readlines()

            file_content = [i[:-1].strip() for i in file_content]
            file_content = [i for i in file_content if i]

            descriptor_df = compute_descriptor(file_content)

            prediction, confidence = predict(descriptor_df)
            confidence_max = [max(i) for i in confidence]

            output_dict_multi = {'compound': file_content,
                        'prediction':[str(int(i)) for i in prediction],
                        'confidence': [str(i) for i in confidence_max]
            }

            output_json_object_multi = json.dumps(output_dict_multi, indent = 4)

            response = app.response_class(
                response=output_json_object_multi,
                status=200,
                mimetype='application/json'
            )
            return response
            '''
    return 'Hello World - multi'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)