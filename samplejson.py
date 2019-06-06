from flask import Flask
from flask import request
  
app = Flask(__name__)
  
@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    content = request.get_json()
    print (content)

    print (content['device'])
    print (content['value'])
    print (content['timestamp'])
    return 'JSON posted'
  
app.run(host='0.0.0.0', port= 8090)