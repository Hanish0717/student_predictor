from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("pass_fail_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # default
    if request.method == 'POST':
        try:
            hours = float(request.form['hours'])
            attendance = float(request.form['attendance'])
            assignments = float(request.form['assignments'])

            prediction = model.predict([[hours, attendance, assignments]])
            result = "Pass" if prediction[0] == 1 else "Fail"
        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
