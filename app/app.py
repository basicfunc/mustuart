from flask import Flask, request, render_template
from backend import predict_genre as process_file

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        result = process_file(file)
        return result
    else:
        return render_template('./index.html')

if __name__ == '__main__':
    app.run()