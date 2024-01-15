from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
import pdb
import torch
from PIL import Image
import wandb
from MLOps_Project.visualizations.visualize_resnet import fig2img



# Baseline Model - Requires only that inheriting object defined self.model and self.loss
class Baseline_Model(LightningModule):
    def __init__(self):
        super().__init__()
        # Define checkpoints and callbacks - Default is save on Epoch End based on max validation accuracy
        # Saved in object, and needs to be forwarded to the Trainer when training
        self.callbacks = [
            ModelCheckpoint(
                dirpath="./models",
                monitor="val/accuracy",
                mode="max",
                filename="LightningTrainedModel2",
                save_on_train_epoch_end=True,
            )
        ]
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, labels = batch
        pred = self.forward(input)
        loss = self.loss(pred, labels)
        self.log("train/loss", loss.item())
        if (batch_idx == 0) and (self.current_epoch > 1):
            self.compute_saliency_map(batch)
        return loss
    
    def compute_saliency_map(self, batch):
        self.model.eval()
        data, labels = batch
        unique_labels = labels.unique()
        L = list(zip(labels.tolist(), range(len(labels))))
        # extract index of the first occurence of each class
        image_idxs = [list(filter(lambda x: x[0] == label, L))[0][1] for label in unique_labels]
        images = data[image_idxs]
        for i, img in enumerate(images):
            img = Variable(img.unsqueeze(0), requires_grad=True)        
            scores = torch.exp(self.forward(img))
            prediction = scores.argmax(dim=1).item()
            class_score = scores.max(dim=1).values.unsqueeze(1)
            class_score.backward()
            saliency = img.grad.data.abs() #torch.max(img.grad.data.abs(),dim=1)
            fig = plt.figure(figsize=(16,8))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            # Show input image
            ax1.imshow(img.squeeze().detach().cpu().numpy())
            ax1.set_title(f'Original image - label {labels[image_idxs][i].item()}')
            ax1.axis('off')
            # Show saliency map
            ax2.imshow(saliency.squeeze().detach().cpu().numpy(), cmap=plt.cm.hot)
            ax2.set_title(f'Saliency map - prediction: {prediction}')
            ax2.axis('off')
            figure = fig2img(fig)
            self.logger.experiment.log({"Saliency figure": wandb.Image(figure)})
            plt.close(fig)
    
    # Default validation step - determines accuracy and loss of validation set
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        pred = self.forward(data)
        # Determine validationa accuracy
        ps = torch.exp(pred)
        _, top_class = ps.topk(1, dim=1)
        correct_guesses = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(correct_guesses.type(torch.FloatTensor))
        # Determine Loss and save both values
        loss = self.loss(pred, labels)
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)
        if(batch_idx == 0):
            print("Validation Loss: " + str(loss.item()))
            print("Validation Accuacy: " + str(accuracy.item()))


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

