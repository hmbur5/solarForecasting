from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)
@app.route("/")
def WelcomePage():
    return 'Hello TRC4200'
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="80", threaded=True, debug=False, use_reloader=False)