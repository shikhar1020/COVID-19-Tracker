from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)

file.close()

@app.route('/', methods = ["GET","POST"])
def hello_world():
    
    if request.method == 'POST':
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        noseRunning = int(myDict['noseRunning'])
        breatingDifficulty = int(myDict['breatingDifficulty'])

        #inference code
        sampleInput = [fever, pain, age, noseRunning, breatingDifficulty]

        OutputPrediction = clf.predict([sampleInput]) #to get the prediction
        print (OutputPrediction)
 
        OutputProbability = clf.predict_proba([sampleInput])[0][1] #to get the probability
        print (OutputProbability)

        return render_template('output.html', inf= round(OutputProbability*100))
    return render_template('index.html')
    #return 'Hello Sangam!' + str(sampleOutputProbability1) + str(sampleOutputPrediction1)

if __name__ == "__main__":
    app.run(debug=True)
