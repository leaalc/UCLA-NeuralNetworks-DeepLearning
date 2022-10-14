import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, constants={'alpha': 1.0, 'dropout': 0.5}):
        super().__init__()

        self.constants = constants

        self.model = nn.Sequential(
            # block 1
            nn.Conv2d(22, 25, kernel_size=(1, 10), padding='same'),
            nn.ELU(alpha=self.constants['alpha']),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(num_features=25),
            nn.Dropout(p=self.constants['dropout']),
            
            # block 2
            nn.Conv2d(25, 50, kernel_size=(1, 10), padding='same'),
            nn.ELU(alpha=self.constants['alpha']),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(num_features=50),
            nn.Dropout(p=self.constants['dropout']),
            
            # block 3
            nn.Conv2d(50, 100, kernel_size=(1, 10), padding='same'),
            nn.ELU(alpha=self.constants['alpha']),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(num_features=100),
            nn.Dropout(p=self.constants['dropout']),
            
            # block 4
            nn.Conv2d(100, 200, kernel_size=(1, 10), padding='same'),
            nn.ELU(alpha=self.constants['alpha']),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(num_features=200),
            nn.Dropout(p=self.constants['dropout']),

        )

        # classification layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4)
        )

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out

class LSTM_Model(nn.Module):
    def __init__(self, constants={'alpha': 1.0, 'dropout': 0.5}):
        super().__init__()

        self.constants = constants

        self.lstm = nn.LSTM(22, 10, 1, batch_first=True, dropout=constants['dropout'])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4)
        )

    def forward(self, x):
        B,C,H,W = x.size()
        x = x.view(B,C,W).permute(0, 2, 1)
        rnn_out, _ = self.lstm(x)
        out = self.fc(rnn_out)
        return out

class CNN_LSTM_Model(CNN_Model):
    def __init__(self, constants={'alpha': 1.0, 'dropout': 0.5}):
        super().__init__(constants=constants)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(100)
        )

        self.lstm = nn.Sequential(
            nn.LSTM(100, 10, 1, batch_first=True, dropout=constants['dropout'])
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10, 4)
        )

    def forward(self, x):
        x = self.model(x)
        fc_out = self.fc1(x)
        B,C = fc_out.size()
        fc_out = fc_out.view(B,1,C)
        lstm_out, _ = self.lstm(fc_out)
        out = self.fc2(lstm_out)
        return out


class GRU_Model(nn.Module):
    def __init__(self, constants={'alpha': 1.0, 'dropout': 0.5}):
        super().__init__()

        self.constants = constants

        self.lstm = nn.LSTM(22, 10, 1, batch_first=True, dropout=constants['dropout'])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4)
        )

    def forward(self, x):
        B,C,H,W = x.size()
        x = x.view(B,C,W).permute(0, 2, 1)
        rnn_out, _ = self.lstm(x)
        out = self.fc(rnn_out)
        return out

class CNN_GRU_Model(CNN_Model):
    def __init__(self, constants={'alpha': 1.0, 'dropout': 0.5}):
        super().__init__(constants=constants)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(100)
        )

        self.lstm = nn.Sequential(
            nn.GRU(100, 10, 1, batch_first=True, dropout=constants['dropout'])
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10, 4)
        )

    def forward(self, x):
        x = self.model(x)
        fc_out = self.fc1(x)
        B,C = fc_out.size()
        fc_out = fc_out.view(B,1,C)
        lstm_out, _ = self.lstm(fc_out)
        out = self.fc2(lstm_out)
        return out