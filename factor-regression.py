import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.pop()
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class ProductDataset(Dataset):
    def __init__(self, targets_file, inputs_file, transform=None, target_transform=None):
        self.targets = pd.read_csv(targets_file, index_col=0)
        self.inputs = pd.read_csv(inputs_file, index_col=0)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_features = self.inputs.iloc[idx, 0:4]
        target = self.targets.iloc[idx, 0:1]
        if self.transform:
            input_features = self.transform(input_features)
        if self.target_transform:
            target = self.target_transform(target)
        return torch.tensor(input_features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def load_datasets(data_file, batch_size: int):
    transf = lambda x: x.drop('date', axis=1) if 'date' in x.columns else x
    m_dataset = ProductDataset(data_file[0], data_file[1])
    m_dataloader = DataLoader(m_dataset, batch_size=batch_size, shuffle=True)
    return m_dataloader

def training_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        ypred = model(x)
        loss = loss_fn(ypred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main(training_data_files, testing_data_files, batch_size, learning_rate, epochs, model_save_path, loss_fn=nn.MSELoss(), optimizer_fn = torch.optim.SGD, dims=(), model_load_path=''):
    training_dataloader = load_datasets(training_data_files, batch_size)
    testing_dataloader = load_datasets(testing_data_files, batch_size)
    print("data loaded")
    if model_load_path:
        model = torch.load(model_load_path)
    else:
        model = NeuralNetwork(dims)
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    # model.to(device)
    print("model loaded")

    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        training_loop(training_dataloader, model, loss_fn, optimizer)
        print(i+1, "th training done")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    torch.save(model, model_save_path)
    torch.save(model.state_dict(), model_save_path.replace(".pth", "_weights.pth"))
    print("model saved")
    return 1


main(training_data_files=("test-targets01.csv", "test-inputs01.csv"),
     testing_data_files=("test-targets02.csv", "test-inputs02.csv"),
     batch_size=5,
     learning_rate=0.1,
     epochs=10,
     model_save_path="models/modeltest01.pth",
     dims=(4,1)
     )
