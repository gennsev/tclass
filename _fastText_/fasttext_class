import fasttext
class fasttext_c:
    def __init__(self, load_model=None, train_data=None, epoch=40, lr=1): 
        if load_model is not None:
            self.model = fasttext.load_model(str(load_model))
        else:
            if train_data is None:
                print('[ERROR] No training data.')
            else:
                self.model = fasttext.train_supervised(input=train_data, epoch=epoch, lr=lr)

    def fasttext_labeler(self, text):
        pred_label = model.predict(text)
        return pred_label
    
    def save_model(self, path):
        self.model.save_model(str(path))
