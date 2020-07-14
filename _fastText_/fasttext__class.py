{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import fasttext\n",
    "class fasttext_c:\n",
    "    def __init__(self, load_model=None, train_data=None, epoch=40, lr=1): \n",
    "        if load_model is not None:\n",
    "            self.model = fasttext.load_model(str(load_model))\n",
    "        else:\n",
    "            if train_data is None:\n",
    "                print('[ERROR] No training data.')\n",
    "            else:\n",
    "                self.model = fasttext.train_supervised(input=train_data, epoch=epoch, lr=lr)\n",
    "\n",
    "    def fasttext_labeler(self, text):\n",
    "        pred_label = model.predict(text)\n",
    "        return pred_label\n",
    "    \n",
    "    def save_model(self, path):\n",
    "        self.model.save_model(str(path))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
