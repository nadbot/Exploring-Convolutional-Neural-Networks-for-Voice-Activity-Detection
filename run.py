from Enums import SNR
from create_data import create_data
from evaluation import calculate_hter, plot_confusion_matrix, predict, average_of_neighbors
from model import create_rgb_model
from train import train

if __name__ == '__main__':
    path_data = './QUT-NOISE-TIMIT/'
    snr = SNR.FIVE
    path_save = "./rgb_spectrograms/"
    x_train, y_train, x_test, y_test, frames, frames_test = create_data(path_data, path_save, snr, recreate=False)

    model = create_rgb_model()
    model = train(model, x_train, y_train, epochs=5)

    # Evaluate
    predictions = predict(model, x_test)
    cf_matrix = plot_confusion_matrix(predictions, y_test, percentage=False)

    hter, mr, far = calculate_hter(cf_matrix)

    print(f'The HTER is {hter}% for SNR {snr.value}, with MR being {mr} and FAR being {far}')

    # Post-processing
    predictions_post_processing = average_of_neighbors(predictions, 5)
    cf_matrix = plot_confusion_matrix(predictions_post_processing, y_test, percentage=False)

    hter, mr, far = calculate_hter(cf_matrix)

    print(f'The post-processed HTER is {hter}% for SNR {snr.value}, with MR being {mr} and FAR being {far}')
