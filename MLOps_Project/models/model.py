from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import timm 
import torch


# Define class for ResNet34 model using the TIMM framework
class ResNet34(LightningModule):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.model = timm.create_model('resnet34', num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()
        # Define checkpoints and callbacks - Default is save on Epoch End based on max validation accuracy
        # Saved in object, and needs to be forwarded to the Trainer when training
        self.callbacks = [
            ModelCheckpoint(
                dirpath="./models",
                monitor="val_accuracy",
                mode="max",
                filename="LightningTrainedModel2",
                save_on_train_epoch_end=True,
            )
        ]
        # Setup first convolutional layer to work with a single channel input
        old_conv_layer = self.model.conv1
        self.model.conv1 = torch.nn.Conv2d(1, old_conv_layer.out_channels, 
                              kernel_size=old_conv_layer.kernel_size, 
                              stride=old_conv_layer.stride, 
                              padding=old_conv_layer.padding, 
                              bias=old_conv_layer.bias)
        if old_conv_layer.in_channels == 3:
            with torch.no_grad():
                self.model.conv1.weight[:,:] = old_conv_layer.weight[:,:].mean(dim=1, keepdim=True)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input, labels = batch
        pred = self.model(input)
        loss = self.loss(pred, labels)
        return loss
    
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
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        if(batch_idx == 0):
            print("Validation Loss: " + str(loss.item()))
            print("Validation Accuacy: " + str(accuracy.item()))
        
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.003)
    
