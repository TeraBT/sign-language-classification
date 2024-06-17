from torchvision import transforms

import utils.training_algorithm
from utils.sign_language_models import SimpleModel
from utils.transformers import simple_transformer


utils.training_algorithm.train('cnn-simple.pth', SimpleLanguageModel, 10, simple_transformer)
