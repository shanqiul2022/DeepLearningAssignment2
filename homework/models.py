"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        self.h = h
        self.w = w
        in_features = 3 * h * w  # 3 channels, RGB
        self.flatten = nn.Flatten(start_dim=1) # flatten all dimensions except batch
        self.linear = nn.Linear(in_features, num_classes) # create a fully connected layer that maps the input to the number of classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x) # flatten the input image
        return self.linear(x) # pass the flattened input through the linear layer


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_size: int = 128,
        num_layers: int = 1, 
        dropout: float = 0.2,
        **_: object,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        self.h = h
        self.w = w
        if hidden_size is not None: hidden_dim = hidden_size
        in_features = 3 * h * w  # 3 channels, RGB

        self.flatten = nn.Flatten(start_dim=1) # flatten all dimensions except batch
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size), # first linear layer
            nn.ReLU(), # activation function
            nn.Dropout(dropout), # dropout for regularization
            nn.Linear(hidden_size, num_classes) # second linear layer to output logits
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x) # flatten the input image
        logits = self.mlp(x)
        return logits # pass the flattened input through the MLP


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 192,     # width of hidden layers 
        num_layers: int = 6,       
        dropout: float = 0.2,
        **_: object,               # ignore extra kwargs gracefully
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        in_features = 3 * h * w  # 3 channels, RGB

        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]  # flatten all dimensions except batch

        # at least one hidden layer
        n_hidden = max(1, num_layers)
        last = in_features
        for _ in range(n_hidden):
            layers += [
                nn.Linear(last, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            last = hidden_dim
        
        #output layer
        layers.append(nn.Linear(last, num_classes))

        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.net(x)

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,     # width of hidden layers (safe for 10 MB cap)
        num_layers: int = 4,       # number of HIDDEN layers (≥4 as required)
        dropout: float = 0.2,
        **_: object,               # ignore extra kwargs gracefully
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        in_features = 3 * h * w  # 3 channels, RGB

        class ResidualBlock(nn.Module):
            def __init__(self, dim: int, dropout: float):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.act = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(dim, dim)

            # He init (good for ReLU)
                nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
                nn.init.zeros_(self.fc1.bias)
                nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
                nn.init.zeros_(self.fc2.bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.fc1(x)
                y = self.act(y)
                y = self.dropout(y)
                y = self.fc2(y)
                y = y + x            # residual add
                y = self.act(y)      # post-activation
                return y
            
        # build the network
        blocks = [nn.Flatten(start_dim=1),
                  nn.Linear(in_features, hidden_dim),
                  nn.ReLU(inplace=True)]  # flatten all dimensions except batch
        
        # stem init
        nn.init.kaiming_normal_(blocks[1].weight, nonlinearity="relu")
        nn.init.zeros_(blocks[1].bias)

         # Residual body
        for _ in range(max(1, num_layers)):
            blocks.append(ResidualBlock(hidden_dim, dropout))

        # Head
        blocks += [nn.Dropout(dropout),
                   nn.Linear(hidden_dim, num_classes)]
        self.net = nn.Sequential(*blocks)

        # Head init
        for m in self.net:
            if isinstance(m, nn.Linear) and m.out_features == num_classes:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
