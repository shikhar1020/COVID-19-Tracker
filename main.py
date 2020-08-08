from flask import Flask, render_template
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)

file.close()

@app.route('/')
def hello_world():
    #inference code
    sampleInput1 = [100.3223, 1, 23, 0, 1]

    sampleOutputPrediction1 = clf.predict([sampleInput1]) #to get the prediction
    print (sampleOutputPrediction1)

    sampleOutputProbability1 = clf.predict_proba([sampleInput1])[0][1] #to get the probability
    print (sampleOutputProbability1)

    return render_template('index.html')
    #return 'Hello Sangam!' + str(sampleOutputProbability1) + str(sampleOutputPrediction1)

if __name__ == "__main__":
    app.run(debug=True)
