import time

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import f1_score

from data_manager import DataManager

import tensorflow as tf
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable.debug_mode()

class Main:
    def __init__(self):
        # experiment results directory
        self.experiment_path = 'results/'

        # seeding
        self.seed = 0
        np.random.seed = self.seed
        tf.seed = self.seed

        # training parameters
        self.epochs = 1
        self.batch_size = 4
        self.validation_split = 0.2

        # data, masks and class labels
        self.dm = None
        self.height = 320
        self.width = 320
        self.augmentation = False
        self.train_file_path = '/content/drive/MyDrive/datasets/dataset'
        self.test_file_path = '/content/drive/MyDrive/datasets/images'

        # compile parameters
        self.learning_rate = 0.001
        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy', 'Recall', 'AUC']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

        # visualization and test parameters
        self.vis_image_path = '/content/drive/MyDrive/datasets/images/Data/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t00_ch00.tif'
        self.vis_patch_size = 128
        self.num_of_patch = 0

    def get_data_and_masks(self, phase='train', augmentation=False, shuffle_data=False):
        path = self.train_file_path
        if (phase == 'test'):    
            path = self.test_file_path

        self.dm = DataManager(height=self.height, width=self.width, path=path)
        data, masks = self.dm.read_data(augmentation=augmentation)
        if (shuffle_data):
            data, masks = self.dm.shuffle_data(data, masks)
        return data, masks

    def train(self, model_obj, load_model=0, model_path='', name='', visualize_stat=True):
        # get data and masks
        data, masks = self.get_data_and_masks(phase='train', augmentation=self.augmentation)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(self.experiment_path + 'best_weights.ckpt', monitor='val_accuracy', mode='max', verbose=1, save_weights_only=True)

        if (load_model):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = model_obj.get_model()
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        history = self.model.fit(
                data, masks, 
                batch_size=self.batch_size, 
                epochs=self.epochs, 
                validation_split=self.validation_split,
                callbacks=[es, mc]
                )
        print('\n\t*** train(): done.')
        if (visualize_stat):
            self.visualize(history, plot_name=name)

    def test(self, model_path='', evaluation=True):
        # get test data and masks
        data, masks = self.get_data_and_masks(phase='test')

        # prediction
        if (model_path == ''):
            pred = self.model.predict(data)
        else:
            self.model.load_weights(model_path)
            pred = self.model.predict(data)

        time_start = time.time()
        self.model(data[0].reshape(1,main_obj.height,main_obj.width,1))
        time_end = time.time()
        print('Prediction time for single image: ', (time_end - time_start))

        print('\n\t*** test(): done.')
        # evaluation
        if (evaluation):
            self.eval(masks, pred)

    def eval(self, masks, pred, name='test_vis.png'):
        threshold = self.get_best_threshold(masks, pred)
        preds_t = (pred > threshold).astype(np.uint8)

        auc = roc_auc_score(masks.reshape(-1), preds_t.reshape(-1))
        acc = accuracy_score(masks.reshape(-1), preds_t.reshape(-1))
        f1_s = f1_score(masks.reshape(-1), preds_t.reshape(-1))
        print('ACC: ',acc,' AUC: ',auc,' F1 Score: ',f1_s,'\nBest threshold: ',threshold)

        # visualize some test images
        fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 30), subplot_kw={'xticks': [], 'yticks': []})
        i, index = 0, 0
        for ax in axs.flat:
            f_s = f1_score(masks[index].reshape(-1), preds_t[index].reshape(-1))
            if (i % 2 == 0):
                ax.imshow(preds_t[index].reshape(self.height, self.width), cmap='gray')
                ax.set_title(str(index) + " prediction f1_score: " + str(f_s))
            else:
                ax.imshow(masks[index].reshape(self.height, self.width), cmap='gray')
                ax.set_title(str(index) + " mask")
                index += 1
            i += 1
        plt.savefig(self.experiment_path + 'eval_imgs_with_scores.png')
        plt.close()
        print('\n\t*** eval(): done.')

    def get_best_threshold(self, masks, pred):
        start, end = 0, 1
        keep_th, gre = 0.5, 0
        e = 1
        while (e > 0.005):
            mid = (start + end) / 2
            score_list = []
            for threshold in [mid*1/2, mid, mid*3/2]:
                preds_t = (pred > threshold).astype(np.uint8)
                score = f1_score(masks.reshape(-1), preds_t.reshape(-1))
                score_list.append(score)
                if (score > gre):
                    gre = score
                    keep_th = threshold

            if (score_list[0] > score_list[2]):
                end = mid
            else:
                start = mid
            e = mid - ((start + end) / 2)
            return keep_th

    def visualize(self, history, plot_name='vis'):
        import pickle
        pickle.dump(history.history['loss'], open(self.experiment_path + 'history.pickle', 'wb'))
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        auc = history.history['auc']
        val_auc = history.history['val_auc']

        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Accuracy')

        plt.subplot(3, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.subplot(3, 1, 3)
        plt.plot(auc, label='Training AUC')
        plt.plot(val_auc, label='Validation AUC')
        plt.legend(loc='upper right')
        plt.ylabel('AUC')
        plt.ylim([0,1.0])
        plt.title('Training and Validation AUC')
        plt.xlabel('epoch')        
        plt.savefig(self.experiment_path + plot_name + '_plot.png') 
        plt.close()

        data = self.dm.read_single_image(self.vis_image_path)
        pred = self.model.predict(data)
        plt.imshow(pred.reshape(self.height, self.width), cmap='gray')
        plt.savefig(self.experiment_path + 'image_' + plot_name + '.png')
        plt.close()

        patch = self.num_of_patch
        pred = pred.reshape(self.height, self.width)
        temp = pred[self.vis_patch_size * patch : (patch+1) * self.vis_patch_size:, self.vis_patch_size * patch : (patch+1) * self.vis_patch_size]
        plt.imshow(temp.reshape(self.vis_patch_size, self.vis_patch_size), cmap='gray')
        plt.savefig(self.experiment_path + 'patch_' + plot_name + '.png')
        plt.close()
        print('\n\t*** visualize(): done.')

if (__name__ == '__main__'):
    pass