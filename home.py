from flask import Flask, render_template
from mrmodel import get_model_api
from textgenrnn import textgenrnn  


model = get_model_api()

# mrmodel = MrModel()

app = Flask(__name__)
def init():
    global model 
    model = textgenrnn(weights_path='zodiac_weights.hdf5',
                        vocab_path='zodiac_vocab.json',
                        config_path='zodiac_config.json')

@app.route("/")
def home():
    return render_template("geminai_web.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("geminai_contact.html")

@app.route("/predict")
def predict():
    # res = mrmodel.generate()
    res = model()
    print(res)    
    # return render_template("zodiac_predict.html", )
    return render_template("zodiac_predict.html", horoscope=res)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
