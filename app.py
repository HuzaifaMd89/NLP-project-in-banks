from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


with open('lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    count_vect = pickle.load(file)

with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

TOPIC_NAMES = {
    0: 'Others',
    1: 'Credit Card Related Services',
    2: 'Mortgage and Loan Services',
    3: 'Bank Account Services',
    4: 'Theft/Dispute'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    complaint = request.form['complaint']
    
  
    X_new_counts = count_vect.transform([complaint])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    
    predicted_class = lr_model.predict(X_new_tfidf)[0]
    topic_name = TOPIC_NAMES[predicted_class]
    
   
    return render_template('result.html', topic=topic_name, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
