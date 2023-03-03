import logging
import os
import sys
import shutil
import csv
import torch


def set_logging(dirname):
    log_format = '%(asctime)s: %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(dirname, 'train_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    for hdlr in logging.getLogger().handlers[:]:  # remove all old handlers
        if isinstance(hdlr, logging.FileHandler):
            print(f"\033[93mWarning: Delete a exist logging handle\033[0m")
            logging.getLogger().removeHandler(hdlr)
    logging.getLogger().addHandler(fh)


class Logger:

    def __init__(self, args, scripts_to_save=None):
        self.args = args

        # Logger directory
        self.exp_name = self.args.save
        self.args.save = f'log/{self.args.save}-{self.args.cluster_type}-Aggr_{self.args.aggr}-' \
                         f'Epochs_{self.args.epochs}-Intervals_{self.args.intervals}-' \
                         f'Layers_{self.args.num_layers}-Embed_{self.args.hidden_channels}-Dropout_{self.args.dropout}'

        self.create_exp_dir(scripts_to_save)
        self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)

        # Log file
        set_logging(self.args.save)

        # Tracking result (Roc-Auc, loss)
        self.results_metric = {'train': [], 'valid': [], 'test': []}
        self.best_epoch = 0
        self.loss = []

    def add_results(self, result, loss):
        self.results_metric['train'].append(result['train']['rocauc'])
        self.results_metric['valid'].append(result['valid']['rocauc'])
        self.results_metric['test'].append(result['test']['rocauc'])
        self.loss.append(loss)

        if self.results_metric['test'][self.best_epoch] < result['test']['rocauc']:
            self.best_epoch = len(self.loss) - 1

    def print_statistic(self):
        logging.info(f"Epoch {len(self.results_metric['train'])} - Results:")
        logging.info(f"Training loss: {self.loss[-1]}")
        logging.info(f"Training ROC-AUC: {self.results_metric['train'][-1]:.4f}")
        logging.info(f"Valid ROC-ACU: {self.results_metric['valid'][-1]:.4f}")
        logging.info(f"Test ROC-ACU: {self.results_metric['test'][-1]:.4f}")

        logging.info(
         f"Best Test ROC-AUC at epoch {self.best_epoch + 1} with: {self.results_metric['test'][self.best_epoch]:.4f}")

    def save_ckpt(self, model, optimizer, loss, epoch, name_pre, name_post='best'):
        """
        Save model, optimizer and loss at a given epoch in save_path with specify name
        :param model: model need to save
        :param optimizer: optimizer used when training model
        :param loss: loss of model at a given epoch
        :param epoch: number of training iteration until model is saved
        :param name_pre: name of model
        :param name_post: type of saving strategy (explain why need to save model?)
        :return:
        """
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        if not os.path.exists(self.args.model_save_path):
            os.mkdir(self.args.model_save_path)
            print(f"Directory {self.args.model_save_path} is created.")

        filename = '{}/{}_{}.pth'.format(self.args.model_save_path, name_pre, name_post)
        torch.save(state, filename)
        print('Model has been saved as {}'.format(filename))

    @staticmethod
    def save_best_result(list_of_dict, file_name, dir_path='best_result'):
        """
        Save multiple metrics into a csv file
        :param list_of_dict: list of metrics
        :param file_name: csv file name to save
        :param dir_path: directory contain file_name, is not exists then create new one
        :return: None
        """
        if not os.path.exists(dir_path):
            print(f"Creating {dir_path} folder...")
            os.mkdir(dir_path)
            print(f"{dir_path} created.")
        csv_file_name = '{}/{}.csv'.format(dir_path, file_name)
        with open(csv_file_name, 'a+') as csv_file:
            csv_writer = csv.writer(csv_file)
            for idx in range(len(list_of_dict)):
                csv_writer.writerow(list_of_dict[idx].values())

    def create_exp_dir(self, scripts_to_save=None):
        """
        Clone dataset, model, training scripts for an experiment
        :param scripts_to_save: all scripts file that need to save
        :return: None
        """
        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)
        else:
            cnt = 2
            while os.path.exists(self.args.save):
                self.args.save = f'log/{self.exp_name}_{cnt}-{self.args.cluster_type}-Aggr_{self.args.aggr}-' \
                                 f'Epochs_{self.args.epochs}-Intervals_{self.args.intervals}-' \
                                 f'Layers_{self.args.num_layers}-Embed_{self.args.hidden_channels}-Dropout_{self.args.dropout}'
                cnt += 1
            os.makedirs(self.args.save)
        print('Experiment dir : {}'.format(self.args.save))

        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.args.save, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.args.save, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
