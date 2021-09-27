import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Neurale Network Klasse entsprechend Pytorch. Erbt von nn.module
    """

    def __init__(self):
        """
        Init methode für die klasse. Hier werden verschiedene Layer für das Netzwerk gebaut
        """
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000 * 1000 * 3, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 4, bias=True)
        )
        self.conv = nn.Sequential(  # Das müsste in etwa unsere idee mit dem 5 kernel abbilden?
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(3, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 4, bias=True)
        )

    def forward(self, x):
        """
        Forward methode entsprechend pytorch. Hier entsteht das eigentliche Model
        :param x:
        :return:
        """
        # x = self.conv(x)
        x = self.conv2(x)
        return x
