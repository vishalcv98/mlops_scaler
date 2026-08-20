from flask import Flask


app = Flask(__name__) # app is now a Flask application instance
# __name__ points the app to the current file 

@app.route("/hello", methods=["GET"])
def hello():
    return "Hello from Flask!"

if __name__ == "__main__":
    app.run(debug=True)