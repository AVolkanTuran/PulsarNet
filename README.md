# PulsarNet
Neural network to detect pulsars through FFA plots.

known_pulsars.txt is a file that contains the list of candidates that are ranked as a pulsar by the Franklin and Marshall NANOStars team. This is used for the training process.

pulsar_net.py is the main file that does the training and testing of the model, and saves the best model into /checkpoints as best_model.pt. You can either train a model, or if there is
already a saved model, you can just apply it on the testing set to see how well the model does.

evaluate_pulsar_net.py is a faster way of seeing how well the model performs on an already classified set. However, this file is mainly useful for
classifying new sets that aren't classified yet using the previously trained model.

S0_json_test_only contains the test set that we used while training our data. It is only 10% of the training data S0.

Installing required packages:
```bash
pip install torch numpy opencv-python scikit-learn tqdm
```
