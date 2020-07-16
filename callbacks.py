import os

from gensim.models.callbacks import CallbackAny2Vec


class CalculateLoss(CallbackAny2Vec):

    def __init__(self, prefix, save_every):
        self.prefix = prefix
        self.save_every = save_every
        self.epoch = 1
        self.previous_loss = 0

    def on_epoch_end(self, model):
        # Hitung selisih epoch
        current_loss = model.get_latest_training_loss() - self.previous_loss
        print("Epoch: #{} - Loss: {}/{}/{}".format(self.epoch, current_loss, model.get_latest_training_loss(),
                                                   self.previous_loss))
        self.previous_loss = model.get_latest_training_loss()

        # Simpan model_arg setiap sekian epoch
        if self.epoch % self.save_every == 0:
            output_path = os.getcwd() + \
                '\\trained_models\\{}_epoch{}.model_arg'.format(
                    self.prefix, self.epoch)
            model.save(output_path)
        # Iterasi counter
        self.epoch += 1
