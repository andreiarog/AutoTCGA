from flask import Flask, render_template, request
import re
import system as s
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/tool")
def tool():
    return render_template("tool.html")

@app.route('/preview',methods = ['POST', 'GET'])
def preview():
	if request.method == 'POST':
		userInput = request.form

		if 'Class-other' in userInput:
			chosen_class = userInput['Class-other']
		else:
			chosen_class = userInput['Class']

		predictor = userInput['Algorithm']
		population = userInput['Population']
		preprocessing = userInput['Preprocessing']
		#dataset = userInput['datasets'][0]
		dataset=userInput['datasets[]']
		user_Features_Include = userInput['user_Features_Include']
		user_Features_Exclude = userInput['user_Features_Exclude']
		inputDf = userInput['inputDf']
		featureselection = userInput['featureselection']
		if 'featureEngineering' in userInput:
			featureEngineering = userInput['featureEngineering']
		else:
			featureEngineering=False

		if (user_Features_Include == ""):
			user_Features_Include=None
		else:
			user_Features_Exclude = re.findall('"([^"]*)"', user_Features_Include)
		
		if (user_Features_Exclude == ""):
			user_Features_Exclude=None
		else:
			user_Features_Exclude = re.findall('"([^"]*)"', user_Features_Exclude)

		if (inputDf == ""):
			inputDf=None	
		else:
			inputDf = pd.DataFrame(eval(inputDf))


		df = s.previewApp(chosen_class, preprocessing = preprocessing, prediction="classification", predictor=predictor, metric="accuracy", remove_FP=True, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall', features_source='TCGA', inputDf=inputDf)

		if isinstance(df, str):
			return render_template("preview.html", error = df, tables = None, chosen_class=chosen_class, preprocessing=preprocessing, predictor=predictor, population=population, inputDf=inputDf, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude)
		else:
			return render_template("preview.html", tables = [df.to_html(classes='data',header="true")], titles=df.columns.values, chosen_class=chosen_class, preprocessing=preprocessing, predictor=predictor, population=population,inputDf=inputDf, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude)

	else:
		return render_template("preview.html")


@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		userInput = request.form

		if 'Class-other' in userInput:
			chosen_class = userInput['Class-other']
		else:
			chosen_class = userInput['Class']
		
		predictor = userInput['Algorithm']
		population = userInput['Population']
		preprocessing = userInput['Preprocessing']
		#dataset = userInput['datasets'][0]
		dataset=userInput['datasets[]']
		user_Features_Include = userInput['user_Features_Include']
		user_Features_Exclude = userInput['user_Features_Exclude']
		inputDf = userInput['inputDf']
		featureselection = userInput['featureselection']
		if 'featureEngineering' in userInput:
			featureEngineering = userInput['featureEngineering']
		else:
			featureEngineering=False

		if (user_Features_Include == ""):
			user_Features_Include=None
		else:
			user_Features_Include = re.findall('"([^"]*)"', user_Features_Include)
		
		if (user_Features_Exclude == ""):
			user_Features_Exclude=None
		else:
			user_Features_Exclude = re.findall('"([^"]*)"', user_Features_Exclude)

		if (inputDf == ""):
			inputDf=None	
		else:
			inputDf = pd.DataFrame(eval(inputDf))

		result, warnings, evaluation = s.createModelApp(chosen_class, preprocessing = preprocessing, prediction="classification", predictor=predictor, metric="accuracy", remove_FP=True, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = None, featureselection=featureselection, featureEngineering=featureEngineering, visualization=True, cancerType = population, features_source=dataset, inputDf=inputDf)
		
		if not warnings :
			warnings=None

		return render_template("result.html",result = result, warnings = warnings, evaluation=evaluation, chosen_class=chosen_class, preprocessing=preprocessing, predictor=predictor, population=population,inputDf=inputDf, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude)
	
	else:
		return render_template("result.html")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)