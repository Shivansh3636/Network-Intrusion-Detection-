# 🛡️ Network Intrusion Detection System

A powerful AI-based network intrusion detection system using machine learning to analyze network traffic and identify potential security threats.

## 🎯 Key Features

- **High Accuracy**: 100% validation accuracy on UNSW-NB15 dataset
- **Real-time Analysis**: Process network traffic data in real-time
- **Modern Web Interface**: Clean, responsive UI with intuitive results
- **Scalable Processing**: Handles large files with chunked processing
- **Detailed Reports**: User-friendly analysis with confidence scores

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 100%
- **Features**: 48 network traffic characteristics
- **Dataset**: UNSW-NB15 (University of New South Wales)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn joblib matplotlib seaborn
```

### 2. Run the Application
```bash
python src/app.py
```

### 3. Access Web Interface
Open your browser and go to: `http://127.0.0.1:5000`

### 4. Validate Model (Optional)
```bash
python src/validate_model.py
```

## 📁 Project Structure

```
network intrusion/
├── src/                    # Source code
│   ├── app.py             # Main Flask application
│   ├── validate_model.py  # Model validation script
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Model training
│   └── debug_preprocessing.py
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── models/                # Trained models and metadata
│   ├── rf_unsw_nb15.joblib    # Main Random Forest model
│   ├── feature_names.npy      # Feature column names
│   └── feature_names.txt      # Human-readable features
├── data/                  # Dataset files
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── [other UNSW-NB15 files]
├── preprocessed/          # Preprocessed data cache
├── logs/                  # Application logs
├── docs/                  # Documentation
└── README.md             # This file
```

## 🔧 Usage

### Web Interface
1. Upload a CSV file containing network traffic data
2. Click "Analyze for Intrusions" 
3. View detailed results with:
   - Risk level assessment
   - Threat detection percentage
   - Detailed sample analysis
   - Confidence scores

### API Endpoints
- `GET /` - Main web interface
- `POST /predict` - Upload and analyze CSV files
- `GET /columns` - View expected data format
- `GET /sample` - Download sample CSV format
- `GET /validate` - View model validation results
- `GET /health` - System health check

### Expected Data Format
The system expects CSV files with network traffic features including:
- Source/destination IPs and ports
- Protocol information
- Connection states and timing
- Data transfer metrics
- 48 total features for optimal analysis

## 🛠️ Features

### Performance Optimizations
- **Chunked Processing**: Handles large files without memory issues
- **Pagination**: Results limited to prevent UI freezing
- **Async Frontend**: Non-blocking file uploads
- **Threaded Backend**: Multiple concurrent requests

### User Experience
- **Modern UI**: Clean, responsive design
- **Real-time Feedback**: Loading indicators and progress
- **Error Handling**: Clear error messages and suggestions
- **Mobile Friendly**: Responsive design for all devices

### Security Analysis
- **Risk Assessment**: Automatic threat level classification
- **Confidence Scores**: Model certainty for each prediction
- **Detailed Reports**: Per-sample analysis with descriptions
- **Alert System**: Visual warnings for high-risk traffic

## 📈 Model Details

The system uses a Random Forest classifier trained on the UNSW-NB15 dataset, which includes:
- 48 network traffic features
- Multiple attack types (DoS, DDoS, Reconnaissance, etc.)
- Both normal and malicious traffic patterns
- Comprehensive feature engineering

## 🔍 Validation Results

Run `python src/validate_model.py` to see detailed performance metrics including:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Per-class performance
- Attack type distribution

## 🤝 Contributing

Feel free to contribute improvements to the model, interface, or documentation!

## 📄 License

This project is for educational and research purposes.
