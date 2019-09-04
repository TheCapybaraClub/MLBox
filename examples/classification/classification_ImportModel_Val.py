"""A classification example using mlbox."""
from mlbox.preprocessing import Reader
from mlbox.preprocessing import Drift_thresholder
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor

import pickle
import pandas as pd

# Paths to the train set and the test set.
paths = ["train_classification.csv", "test_classification.csv"]
# Name of the feature to predict.
# This columns should only be present in the train set.
target_name = "Survived"

# Reading and cleaning all files
# Declare a reader for csv files
rd = Reader(sep=',')
# Return a dictionnary containing three entries
# dict["train"] contains training samples withtout target columns
# dict["test"] contains testing elements withtout target columns
# dict["target"] contains target columns for training samples.
data = rd.train_test_split(paths, target_name)

dft = Drift_thresholder()
df = dft.fit_transform(data)

# Make prediction and save the results in save folder.
#prd = Predictor()
#prd.fit_predict(best, data)

to_path = 'save'
if (df['target'].dtype == 'int'):
    enc_name = "target_encoder.obj"
    pipeline_name = "scoring_pipeline.obj"

    #enc
    try:

        fhand = open(to_path + "/" + enc_name, 'rb')
        enc = pickle.load(fhand)
        fhand.close()

    except:
        ValueError("Unable to load '" + enc_name +
                         "' from directory : " + to_path)

    #pipeline
    try:

        fhand = open(to_path + "/" + pipeline_name, 'rb')
        pp = pickle.load(fhand)
        fhand.close()
        print("saved scoring pipeline object found, predicting using previously trained object...")

    except:
        #if the file is not found, maybe it hasn't been created yet
        #Is this the first training run? 
        print("no saved scoring pipeline object found, predicting using newly trained object...")
        pass

    try:
        if(True):
            print("")
            print("predicting ...")

        pred = pd.DataFrame(pp.predict_proba(df['test']),
                            columns=enc.inverse_transform(range(len(enc.classes_))),
                            index=df['test'].index)
        pred[df['target'].name + "_predicted"] = pred.idxmax(axis=1)  # noqa

        try:
            pred[df['target'].name + "_predicted"] = pred[df['target'].name + "_predicted"].apply(int)  # noqa
        except:
            pass

    except:
        ValueError("Can not predict")
else:
    print("df target data type bad")


if(True):
    print("")
    print("dumping predictions into directory : "+self.to_path + " ...")

pred.to_csv(to_path
            + "/"
            + df['target'].name
            + "_predictions.csv",
            index=True)





















