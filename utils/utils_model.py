import numpy as np

from .utils_read_dataset import *


def get_prediction_emotion(xgb_model, input_name, embedding, label_encoder):
    """
    This function take input as name and return its predicted emotion
    """
    processed_name = sentence_to_vector(input_name, embedding)
    processed_name = np.expand_dims(processed_name, axis=0)

    y_pred_label = xgb_model.predict(processed_name)  # Inference

    # convert from label (187) to emotion unicode -> (U+1F913)
    y_pred_unicode = label_encoder.inverse_transform(y_pred_label)[0]
    y_pred_emotion = convert_unicode_2_emoji(y_pred_unicode)

    return y_pred_emotion


def get_top_k_prediction(xgb_model, input_name, top_k, embedding, label_encoder):
    """
    This function take input as text and return its TOP-K predicted emotion
    """    

    # Add batch dimension for input text
    processed_name = sentence_to_vector(input_name, embedding)
    processed_name = np.expand_dims(processed_name, axis=0)

    # Get top_k predicted probability
    y_pred_proba = xgb_model.predict_proba(processed_name)[0]
    top_k_y_pred_label = np.argsort(y_pred_proba)[-top_k:][::-1]  # Just a trick to reverse an array

    # Loop through all top_k predicted 
    list_predcited_emotion = []
    for y_pred_label in top_k_y_pred_label:
        y_pred_unicode = label_encoder.inverse_transform([y_pred_label])[0]
        y_pred_emotion = convert_unicode_2_emoji(y_pred_unicode)
        list_predcited_emotion.append(y_pred_emotion)
    
    return list_predcited_emotion


def calculate_top_k_accuracy(model, X_test, y_test, top_k=10):
    y_pred_proba = model.predict_proba(X_test)

    # Get the indices of the top k predictions
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -top_k:]

    top_k_correct = [y_test[i] in top_k_preds[i] for i in range(len(y_test))]

    top_k_accuracy = np.mean(top_k_correct)
    return top_k_accuracy