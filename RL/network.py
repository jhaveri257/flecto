import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.1):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout1d(dropout_prob))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout1d(dropout_prob))

        #encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_sizes[-1], nhead=64)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:  # Apply ReLU to all layers except the last one
                x = layer(x)

        #x = self.transformer(x)
        x = self.layers[-1](x)
        
        return x
    
if __name__ == "__main__":
    
    input_size = 54
    hidden_sizes = [256,256]
    output_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(input_size, hidden_sizes, output_size).to(device)
    print(model)

    lowest_loss = float('inf')

    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=250, gamma=0.5)

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.normpath(os.path.join(script_dir, "..", "stats_reno_aioquic.csv"))
    columns_to_remove = ["timestamp"]
    batch_size = 128

    # Create custom dataset
    custom_dataset = CustomDataset(csv_file, columns_to_remove)

    train_size = int(0.99 * len(custom_dataset))  # 80% for training, adjust as needed
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model.load_state_dict(torch.load('model.pth'))

    #model.eval()
    #with torch.no_grad():
    #    output = model(custom_dataset.data.to(device))
    #difference = torch.abs(custom_dataset.get_original_y().to(device) - custom_dataset.get_original_from_output(output))
    #print(difference.mean().item())

    num_epochs = 50
    loss_values = []
    loss_values_test = []

    progress_bar = tqdm(total=num_epochs, desc=f"Training", position=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)            
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

        epoch_loss = total_loss/len(train_dataloader)
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {(epoch_loss):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        loss_values.append(total_loss/len(train_dataloader))
        progress_bar.update(1)  # Update progress bar
        scheduler.step()
        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(script_dir, 'model.pth'))

        total_loss_test = 0

        model.eval()
        for input, labels in test_dataloader:
            input = input.to(device)
            labels = labels.to(device)
            output = model(input)
            loss = criterion(output, labels)
            total_loss_test += loss.item()

        loss_values_test.append(total_loss_test/len(test_dataloader))
        
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.plot(loss_values_test, label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(script_dir, "net_train.png"))
    print(f'Final Loss: {lowest_loss:.4f}')