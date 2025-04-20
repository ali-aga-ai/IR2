from flask import Flask , request
from query import query
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all origins by default

api_key = "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"

@app.route("/respond", methods=['POST'])
def respond():

    data = request.get_json(force=True)  # This works with POST
    userQuery = data.get('message')
    response = query(userQuery, api_key)
    print(response[-1])
    return response[-1]['content']

if __name__ == '__main__':
    app.run(debug=True)