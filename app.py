from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import requests
import pandas as pd
import json

from config import ModelConfig
from data_processor import DataProcessor
from gradient import GradientDescent
from predictor import WineQualityPredictor


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    config = ModelConfig()
    data_processor = DataProcessor(config)
    X_train, X_test, y_train, y_test = data_processor.load_and_preprocess_data()
    predictor = WineQualityPredictor(config)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/modules/gradient', methods=['POST'])
    def run_gradient():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            params = {
                'modo': data.get('modo', 'batch'),
                'eta': float(data.get('eta', 0.1)),
                'n_iterations': int(data.get('n_iterations', 1000)),
                'batch_size': int(data.get('batch_size', 16))
            }
            
            if params['modo'] not in ['batch', 'stochastic', 'mini-batch']:
                return jsonify({'error': 'Invalid mode'}), 400
            
            gd = GradientDescent(
                X_train=X_train,
                y_train=y_train,
                eta=params['eta'],
                n_iterations=params['n_iterations'],
                batch_size=params['batch_size']
            )
            
            # Get training results
            theta, plot_data = gd.gradiente(modo=params['modo'])
            
            # Convert numpy array to list for JSON serialization
            theta_list = theta.tolist()
            
            response_data = {
                'status': 'success',
                'parameters': params,
                'results': {
                    'theta': theta_list,
                    'plot_data': plot_data
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
            
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data or 'features' not in data:
                return jsonify({'error': 'Features are required'}), 400
            
            predictions = predictor.predict_all(data['features'])
            
            return jsonify({
                'status': 'success',
                'predictions': predictions
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)