import os

from gensim.models.callbacks import CallbackAny2Vec


class CalculateLoss(CallbackAny2Vec):
    """Save model after several epochs."""

    def __init__(self, path_prefix, save_every):
        self.path_prefix = path_prefix
        self.save_every = save_every
        self.epoch = 0
        self.previous_loss = 0

    def on_epoch_end(self, model):
        # Get loss difference
        current_loss = model.get_latest_training_loss() - self.previous_loss
        print("Epoch: #{} - Loss: {}/{}/{}".format(self.epoch, current_loss, model.get_latest_training_loss(), self.previous_loss))
        # Update 'previous loss'
        self.previous_loss = current_loss

        # Save model every set epoch
        if self.epoch % self.save_every == 0:
            output_path = os.getcwd() + '\\saved_models\\{}_epoch{}.model'.format(self.path_prefix, self.epoch)
            model.save(output_path)
        # Iterate epoch counter
        self.epoch += 1
