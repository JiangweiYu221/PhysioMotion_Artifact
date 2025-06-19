# PhysioMotion_Artifact
Code for motion artifact collection, preprocessing and labeling  
Checking_Labels.py provides the code to generate a signal plot and highlights the artifact signals.  
Labeling_System.py provides the tool to annotate the artifact contaminated signals at the channel level.  
preprocess.py is the pipeline for preprocessing.  
sample.csv is the default channel option for different types of artifact.
  
In classification folder:  
Preprocess_and_Segment.py is the preprocessing pipeline before training.  
Binary_classification.py provides the model for binary classification.  
Multiclass_classification.py provides the model for multiclass classification.  
