from fastai.learner import load_learner, Path
import os

class Classifier:
    def __init__(self) -> None:
        fruit = os.path.join('.', 'fruit-v1.pkl')
        self.__fruit = load_learner(fruit)
        self.__avocado = load_learner(fruit)

    def __get_model(self, model):
        if model == 'fruit':
            return self.__fruit
        elif model == 'avocado':
            return self.__avocado
        else:
            return None

    def get_prediction(self, filepath, model):
        model = self.__get_model(model)
        prediction = model.predict(filepath)
        Path(filepath).unlink()
        return prediction
