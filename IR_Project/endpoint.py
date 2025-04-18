from flask import Flask , request
from query import query
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all origins by default

api_key = ""

@app.route("/respond", methods=['POST'])
def respond():

    data = request.get_json(force=True)  # This works with POST
    userQuery = data.get('message')
    response = query(userQuery, api_key)
    print(response[-1]['content'])
    return response[-1]['content']

if __name__ == '__main__':
    app.run(debug=True)