import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt

EPS = 1e-10


# Data preprocessing module
def load_and_preprocess_data(data_path, sample_size=10000, random_seed=42):
    """
    Load and preprocess the data.

    Args:
        data_path (str): Path to the data file.
        sample_size (int): Number of samples to use.
        random_seed (int): Random seed for reproducibility.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Preprocessed data splits.
    """
    data = pd.read_csv(data_path)
    data = data.sample(n=sample_size, random_state=random_seed)
    df = pd.DataFrame(data)

    categorical_features = df.select_dtypes(include=["object", 'bool']).columns
    for feature in categorical_features:
        df[feature] = df[feature].astype('category').cat.codes

    target = df['LoanStatus'].values
    features = df.drop('LoanStatus', axis=1)

    scaler = StandardScaler()
    features[features.columns] = scaler.fit_transform(features[features.columns])

    correlation_matrix = features.corr()
    correlation_threshold = 0.6
    highly_correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_features.add(colname)

    selected_features = [feature for feature in features.columns if feature not in highly_correlated_features]
    features = features[selected_features]

    X_train, X_temp, y_train, y_temp = train_test_split(features.values, target, test_size=0.2,
                                                        random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)
    return X_train, y_train, X_val, y_val, X_test, y_test


# Element-wise gating module
class ElementWiseGating(nn.Module):
    """
    Element-wise gating module for the VAE.

    Args:
        latent_dim (int): Dimension of the latent space.
        output_dim (int): Dimension of the output.
        num_experts (int): Number of experts.
    """

    def __init__(self, latent_dim, output_dim, num_experts=8):
        super(ElementWiseGating, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(self.latent_dim, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, self.output_dim, bias=True)
        ) for _ in range(self.num_experts)])
        self.gating = nn.Linear(self.latent_dim, self.num_experts * self.output_dim)
        self.dimension_weights = []

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        gate_output = self.gating(x)
        gate_output = gate_output.view(x.size(0), self.num_experts, self.output_dim)
        gate_output = F.softmax(gate_output, dim=1)
        output = torch.einsum('boe,beo->bo', expert_outputs, gate_output)
        return output


# VAE Model
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Args:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent space.
        output_dim (int): Dimension of the output.
        num_experts (int): Number of experts for the gating mechanism.
        gating_type (str): Type of gating mechanism.
    """

    def __init__(self, input_dim, latent_dim, output_dim=2, num_experts=8, gating_type='element_wise'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.likelihood = 'other'
        self.act = nn.LeakyReLU()

        self.encoder_input_layer = nn.Linear(input_dim, 256, bias=True)
        self.encoder_hidden_layer = nn.Linear(256, 2 * latent_dim, bias=True)

        self.fc_mu = nn.Linear(2 * latent_dim, latent_dim)
        self.fc_var = nn.Linear(2 * latent_dim, latent_dim)

        self.gating = ElementWiseGating(latent_dim, input_dim, num_experts)
        self.rnn = nn.GRU(latent_dim, latent_dim, num_layers=2, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, input_dim, bias=True),
        )

        self.fc_lr = nn.Linear(input_dim, output_dim, bias=True)

    def encode(self, x):
        x = F.normalize(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.act(self.encoder_input_layer(x))
        x = self.act(self.encoder_hidden_layer(x))
        mu = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.gating(z)
        z = self.act(z)
        h = self.decoder(z)
        return h

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        h = self.decode(z)
        return h, mu, log_var


# VAE Loss Function
def vae_loss(recon_x, x, mu, log_var):
    """
    Compute the VAE loss.

    Args:
        recon_x (torch.Tensor): Reconstructed data.
        x (torch.Tensor): Original data.
        mu (torch.Tensor): Mean of the latent distribution.
        log_var (torch.Tensor): Log variance of the latent distribution.

    Returns:
        loss (torch.Tensor): VAE loss.
    """
    recons_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return torch.mean(recons_loss + kld)


# VAE Training Function
def train_vae(model, train_loader, val_loader, epochs=10, lr=0.0001):
    """
    Train the VAE model.

    Args:
        model (VAE): VAE model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.

    Returns:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            recon_x, mu, log_var = model(data)
            loss = vae_loss(recon_x, data, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader.dataset))
        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                recon_x, mu, log_var = model(data)
                loss = vae_loss(recon_x, data, mu, log_var)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader.dataset))
        print(f'Epoch {epoch + 1}, Val Loss: {val_losses[-1]}')
    return train_losses, val_losses


# Train one fold
def train_one_fold(seed=42):
    """
    Train one fold of the model.

    Args:
        seed (int): Random seed for reproducibility.
    """
    data_path = 'path'
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(data_path, random_seed=seed)

    input_dim = X_train.shape[1]
    latent_dim = 32
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, gating_type='element_wise')
    vae = vae.cuda()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    train_losses, val_losses = train_vae(vae, train_loader, val_loader, epochs=10)

    plt.plot(train_losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.show()

    vae.eval()
    with torch.no_grad():
        recon_x = []
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            recon_x_, _, _ = vae(data)
            recon_x += list(recon_x_.cpu().numpy())

        rf = RandomForestClassifier()
        rf.fit(recon_x, y_train)

        recon_x = []
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            recon_x_, _, _ = vae(data)
            recon_x += list(recon_x_.cpu().numpy())
        y_pred = rf.predict(recon_x)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'ROC AUC: {roc_auc}')


if __name__ == '__main__':
    train_one_fold()