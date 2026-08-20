from flask import Flask, request

app = Flask(__name__)

@app.route("/hello", methods=["GET", "POST"])
def hello():

    if request.method == "GET":
        return "Hello from GET request"

    elif request.method == "POST":

        data = request.json
        name = data["name"]

        return f"Hello {name} from POST request"

if __name__ == "__main__":
    app.run(debug=True)