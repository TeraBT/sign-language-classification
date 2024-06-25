import utils.training_algorithm
from utils.sign_language_models import ImprovedModel
from utils.transformers import improved_transformer

utils.training_algorithm.train('cnn-improved.pth', ImprovedModel, 20, improved_transformer)
