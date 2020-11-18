import torch
import torch.optim.lr_scheduler
from flair.training_utils import store_embeddings
import os
from .data import DataLoader
import src.logger as logger

class trainer:
    def __init__(self, model, corpus, optimizer_method, base_path):
        self.model = model
        self.corpus = corpus
        self.optimizer_method = optimizer_method
        self.history = {
            'train_loss': [], 'train_results': [],
            'dev_loss':   [], 'dev_results':   [],
            'test_loss':  [], 'test_results':  [],
            'learning_rate': []
        }
        self.epoch = 0
        self.base_path = base_path
        self._checkpoints_dir = os.path.join(base_path, 'checkpoints')
        self._history_dir = os.path.join(base_path, 'history')
        if not os.path.exists(self._checkpoints_dir):
            os.makedirs(self._checkpoints_dir)
        if not os.path.exists(self._history_dir):
            os.makedirs(self._history_dir)

    def train(self, n_epochs,
                    train_with_dev=False, mixup_training=False, embedding_storage_mode='cpu',
                    batch_size=32, shuffle=True, num_workers=0, n_passes=2,
                    learning_rate=0.1, learning_rate_decay=0.5, learning_rate_min=1e-4, patience=3, step_size=0,
                    monitor_train=False, monitor_dev=False, monitor_test=False, print_every_batch=100,
                    save_every_epoch=0, checkpoint_name=None,
                    ):
        logger.log('Model:')
        logger.log(str(self.model))
        logger.log('Trainer parameters:')
        logger.log(f'  -n_epochs={n_epochs}')
        logger.log(f'  -train_with_dev={train_with_dev}')
        logger.log(f'  -mixup_training={mixup_training}')
        logger.log(f'  -embedding_storage_mode={embedding_storage_mode}')
        logger.log(f'  -batch_size={batch_size}')
        if not mixup_training:
            logger.log(f'  -shuffle={shuffle}')
        else:
            logger.log(f'  -n_passses={n_passes}')
        logger.log(f'  -learning_rate={learning_rate}')
        logger.log(f'  -learning_rate_decay={learning_rate_decay}')
        logger.log(f'  -learning_rate_min={learning_rate_min}')
        if step_size == 0:
            logger.log(f'  -patience={patience}')
        else:
            logger.log(f'  -step_size={step_size}')

        train_data = self.corpus.train

        if train_with_dev:
            train_data = self.corpus.train + self.corpus.dev

        self.optimizer = self.optimizer_method(self.model.parameters(), lr=learning_rate)

        if patience == 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, learning_rate_decay)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min" if train_with_dev else "max", patience=patience,
                factor=learning_rate_decay, verbose=False)

        if checkpoint_name is not None:
            self.load(checkpoint_name)

        if mixup_training:
            train_loader = DataLoader(train_data,
                                      batch_size=2*batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
        else:
            train_loader = DataLoader(train_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers)

        for epoch in range(self.epoch, n_epochs):
            if learning_rate < learning_rate_min:
                logger.log('Quitting training: learning rate is to small!')
                break
            self.model.train()
            logger.log(f'Epoch ({epoch+1}/{n_epochs}), lr={learning_rate}')
            if mixup_training:
                train_loss = self._train_one_epoch_mixup(train_loader, n_passes, embedding_storage_mode, print_every_batch)
            else:
                train_loss = self._train_one_epoch_normal(train_loader, embedding_storage_mode, print_every_batch)

            if not monitor_train:
                logger.log(f'Train loss: {train_loss:.4f}')
                self.history['train_loss'].append(train_loss)

            self.model.eval()
            eval_score = 0
            if monitor_train:
                loss, result = self.model.evaluate(DataLoader(self.corpus.train,
                                                              batch_size=batch_size,
                                                              num_workers=num_workers),
                                                   embedding_storage_mode)
                logger.log('Train evaluation:')
                logger.log(f'  loss: {loss:.4f}')
                for metric in result.metrics.keys():
                    logger.log(f'  {metric}: {result.metrics[metric].__str__()}')

                self.history['train_loss'].append(loss)
                self.history['train_results'].append(result)

            if monitor_dev or not train_with_dev:
                loss, result = self.model.evaluate(DataLoader(self.corpus.dev,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers),
                                                  embedding_storage_mode)
                eval_score = result.main_score
                if monitor_dev:
                    logger.log('Dev evaluation:')
                    logger.log(f'  loss: {loss:.4f}')
                    for metric in result.metrics.keys():
                        logger.log(f'  {metric}: {result.metrics[metric].__str__()}')
                    self.history['dev_loss'].append(loss)
                    self.history['dev_results'].append(result)

            if monitor_test:
                loss, result = self.model.evaluate(DataLoader(self.corpus.test,
                                                              batch_size=batch_size,
                                                              num_workers=num_workers),
                                                   embedding_storage_mode)
                logger.log('Test evaluation:')
                logger.log(f'  loss: {loss:.4f}')
                for metric in result.metrics.keys():
                    logger.log(f'  {metric}: {result.metrics[metric].__str__()}')
                self.history['test_loss'].append(loss)
                self.history['test_results'].append(result)

            for group in self.optimizer.param_groups:
                learning_rate = group['lr']

            self.history['learning_rate'].append(learning_rate)
            if step_size == 0:
                self.scheduler.step(train_loss if train_with_dev else eval_score)
            else:
                self.scheduler.step()
            self.epoch += 1

            if save_every_epoch != 0 and (epoch % save_every_epoch == 0):
                _path = self.save(epoch)
                logger.log(f'Epoch {epoch} saved to {_path}')

            with open(os.path.join(self._history_dir, 'monitor.txt'), 'a') as f:
                f.write(f'epoch {epoch} => learning rate: {self.history["learning_rate"][-1]:.4f}  train loss: {self.history["train_loss"][-1]:.4f}')
                if self.history['train_results']:
                    f.write(f'  train score: {self.history["train_results"][-1].main_score:.4f}') 
                if self.history['dev_loss']:
                    f.write(f'  dev loss: {self.history["dev_loss"][-1]:.4f}')

                if self.history['dev_results']:
                    f.write(f'  dev score: {self.history["dev_results"][-1].main_score:.4f}')

                if self.history['test_loss']:
                    f.write(f'  test loss: {self.history["test_loss"][-1]:.4f}')

                if self.history['test_results']:
                    f.write(f'  test score: {self.history["test_results"][-1].main_score:.4f}')
                f.write('\n')
        if self.corpus.test:
            loss, result = self.model.evaluate(DataLoader(self.corpus.test,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers),
                                            embedding_storage_mode)
            logger.log('Final evaluation on test:\n')
            logger.log(f'  loss: {loss:.4f}')
            for metric in result.metrics.keys():
                logger.log(f'  {metric}: {result.metrics[metric].__str__()}')

        _path = os.path.join(self.base_path, 'final-model.pt')
        torch.save(self.model.state_dict(), _path)
        logger.log(f'Final model saved to {_path}')

        _path = os.path.join(self._history_dir, 'history.pt')
        torch.save(self.history, _path)
        logger.log(f'History saved to {_path}')

        return self.history

    def _train_one_epoch_normal(self, train_loader, embedding_storage_mode, print_every_batch):
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch_loss = self.model.forward_loss(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            store_embeddings(batch[0], embedding_storage_mode)
            epoch_loss += batch_loss.item()
            if (i % print_every_batch == 0 or i+1 == len(train_loader)):
                logger.log(f'  Batch ({i+1}/{len(train_loader)}), loss: {epoch_loss/(i+1):.4f}')
        return epoch_loss / len(train_loader)

    def _train_one_epoch_mixup(self, train_loader, n_passes, embedding_storage_mode, print_every_batch):
        epoch_loss = 0
        for j in range(n_passes):
            logger.log(f'  Pass ({j+1}/{n_passes})')
            pass_loss = 0
            for i, big_batch in enumerate(train_loader):
                sz = len(big_batch[0]) // 2
                batch1 = (big_batch[0][:sz], big_batch[1][:sz])
                batch2 = (big_batch[0][sz:sz+sz], big_batch[1][sz:sz+sz])
                self.optimizer.zero_grad()
                batch_loss = self.model.forward_loss(batch1, batch2)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                store_embeddings(batch1[0], embedding_storage_mode)
                store_embeddings(batch2[0], embedding_storage_mode)
                pass_loss += batch_loss.item()
                if (i % print_every_batch == 0 or i+1 == len(train_loader)):
                    logger.log(f'    Batch ({i+1}/{len(train_loader)}), loss: {pass_loss/(i+1):.4f}')
            epoch_loss += pass_loss
        return epoch_loss / (len(train_loader) * n_passes)


    def save(self, epoch):
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(state, os.path.join(self._checkpoints_dir, f'checkpoint-{epoch}.ckpt'))
        return f'checkpoint-{epoch}.ckpt'

    def load(self, checkpoint):
        state = torch.load(os.path.join(self._checkpoints_dir, checkpoint))
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.history = state['history']
        self.epoch = state['epoch']


