import fasttext
import os
import pandas as pd


class fasttextClassifier:
    def __init__(self, load_model=None, train_data=None, epoch=40, lr=1): 
        if load_model is not None:
            self.model = fasttext.load_model(str(load_model))
        else:
            if train_data is None:
                print('[ERROR] No training data.')
            else:
                self.model = fasttext.train_supervised(input=train_data, epoch=epoch, lr=lr)

    def fasttext_labeler(self, text):
        pred_label = self.model.predict(text)
        return pred_label
    
    def save_model(self, path):
        self.model.save_model(str(path))

    def gen_training(self, path, label, outfile_path):
        """
        Parameters:
            path:
            label:
            outfile_path:
             
        """   
        rows_list = []
        for filename in os.listdir(path):
            with open(f'{path}/{filename}', encoding='utf-8') as f:
                text = f.read()
                text_composed = text.replace('\n', ' ')
                ft_label = '__label__' + str(label)
                ft_content = str(text_composed) 
                dict1 = {
                    'label'   : ft_label,
                    'content' : ft_content
                 }

            rows_list.append(dict1)
            
        trainingDf = pd.DataFrame(rows_list)   
        trainingDf.to_csv(outfile_path, header=False, index=False, sep='\t')
        return trainingDf.head()
    
    def fasttext_test(self, testfilepath):
        test_result = self.model.test(testfilepath)
        return test_result
    