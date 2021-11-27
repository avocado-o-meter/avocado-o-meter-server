from fastai.learner import load_learner, Path
import urllib.request
class Classifier:
    def __init__(self) -> None:
        urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1qYuNCaQoZDuDCX_llHgriQn2qWE62KyA', 'fruit-v1.pkl')
        self.__fruit = load_learner(Path('.'), 'fruit-v1.pkl')
        self.__avocado = load_learner(Path('.'), 'fruit-v1.pkl')

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
