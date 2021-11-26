from fastai.learner import load_learner, Path

class Classifier:
    def __init__(self) -> None:
        path = Path('./models/')
        self.__fruit = load_learner(path/'fruit-v1.pkl')
        self.__avocado = load_learner(path/'fruit-v1.pkl')

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
