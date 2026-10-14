
import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """Experimental CNN for time-series posture features."""

    def __init__(self, input_features=4, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                input_features, 32,
                kernel_size=5, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                32, 64,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                64, 128,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = TemporalCNN().to(device)
    sample = torch.randn(2, 64, 4, device=device)

    # Convert (batch, time, features) to (batch, features, time).
    sample = sample.transpose(1, 2)

    with torch.no_grad():
        output = model(sample)

    print("Device:", device)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    ))
    print("Architecture forward pass: successful")
    print("No training performed; no labels were used.")


if __name__ == "__main__":
    main()
