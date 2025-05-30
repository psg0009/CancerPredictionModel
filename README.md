# Blood Cancer Prediction Model

This project implements a deep learning model for blood cancer prediction using both image and genomic data. The model combines computer vision techniques with genomic analysis to provide accurate predictions.

## Features

- WBC (White Blood Cell) segmentation
- Multi-modal learning (image + genomic data)
- Multiple class balance strategies
- Comprehensive metrics and visualizations
- Data augmentation techniques
- Imbalance handling strategies

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- OpenCV
- scikit-learn
- albumentations
- imbalanced-learn
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/psg0009/CancerPredictionModel.git
cd CancerPredictionModel
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python a1.py
```

The script will:
1. Generate synthetic data with specified class balance ratios
2. Train models using different imbalance handling strategies
3. Generate visualizations and metrics
4. Save results in the experiments directory

## Project Structure

- `a1.py`: Main script containing model architecture and training logic
- `training_utils.py`: Training utilities and metrics calculation
- `visualization_utils.py`: Visualization functions
- `requirements.txt`: Project dependencies
- `experiments/`: Directory for storing results and models

## License

This project is licensed under the MIT License - see the LICENSE file for details. 