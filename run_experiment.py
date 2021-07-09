from main import Main
from models.unet import Unet
from models.u2net import U2net
from loss_functions import LossFunctions

def experiment(main_obj, model_obj, loss, name):
    experiment_name = name
    main_obj.experiment_path = 'results/' + experiment_name + '/'

    main_obj.loss = loss
    main_obj.train(model_obj, load_model=0, model_path='', name=experiment_name)
    main_obj.test(model_path=main_obj.experiment_path + 'best_weights.ckpt', evaluation=True)

if (__name__ == '__main__'):
    loss_obj = LossFunctions()
    main_obj = Main()

    # seed
    main_obj.seed = 0

    # training parameters
    main_obj.epochs = 100
    main_obj.batch_size = 4
    main_obj.validation_split = 0.3
    main_obj.learning_rate = 0.001

    # data, masks
    main_obj.augmentation = False
    main_obj.train_file_path = 'ConferenceVideoSegmentationDataset/train'
    main_obj.test_file_path = 'ConferenceVideoSegmentationDataset/test'

    # Experiment 1 - U-Net + Binary Cross-Entropy Loss
    model_obj = Unet(height=main_obj.height, width=main_obj.width, seed=main_obj.seed)
    experiment(main_obj, model_obj, 'binary_crossentropy', 'unet_bce')
    # Experiment 2 - U-Net + Focal Loss
    model_obj = Unet(height=main_obj.height, width=main_obj.width, seed=main_obj.seed)
    experiment(main_obj, model_obj, loss_obj.focal_loss, 'unet_focal')
    # Experiment 3 - U2-Net + Binary Cross-Entropy Loss
    model_obj = U2net(height=main_obj.height, width=main_obj.width, seed=main_obj.seed)
    experiment(main_obj, model_obj, 'binary_crossentropy', 'u2net_focal')
    # Experiment 4 - U2-Net + Focal Loss
    model_obj = U2net(height=main_obj.height, width=main_obj.width, seed=main_obj.seed)
    experiment(main_obj, model_obj, loss_obj.focal_loss, 'u2net_focal')
