import sys
import faiss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F

def cosine_loss(pred, target):
    cos_sim = F.cosine_similarity(pred, target, dim=1)  # → shape [batch_size]
    loss = 1 - cos_sim.mean()
    return loss

class REBELtoRDF2VecMapper(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=768, output_dim=100, num_blocks=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x

def visualize_spaces(rdf2vec_embs, rebel_embs, model, sample_size=200, title="Embedding Spaces (t-SNE)"):
    # Sample a subset for visualization (t-SNE is slow on large sets)
    idx = np.random.choice(len(rdf2vec_embs), sample_size, replace=False)
    rdf2vec_subset = np.array(rdf2vec_embs)[idx]
    rebel_subset = np.array(rebel_embs)[idx]

    # Project REBEL embeddings through your model
    model.eval()
    with torch.no_grad():
        rebel_tensor = torch.tensor(rebel_subset).to(next(model.parameters()).device)
        projected_subset = model(rebel_tensor).cpu().numpy()

    # Stack all vectors for t-SNE
    combined = np.vstack([rdf2vec_subset, projected_subset])

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=30)
    reduced = tsne.fit_transform(combined)

    # Split the reduced 2D space
    rdf2vec_2d = reduced[:sample_size]
    projected_2d = reduced[sample_size:]

    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(rdf2vec_2d[:, 0], rdf2vec_2d[:, 1], c='blue', label='RDF2Vec (target)', alpha=0.6)
    plt.scatter(projected_2d[:, 0], projected_2d[:, 1], c='red', label='REBEL → RDF2Vec (projected)', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("embedding_spaces_tsne.png")

def align_embeddings(rebel_embeddings_file: str, rdf2vec_embeddings_file: str):
    rebel_index = faiss.read_index(rebel_embeddings_file)
    rebel_embeddings = [rebel_index.reconstruct(i) for i in range(rebel_index.ntotal)]
    print(f"read {len(rebel_embeddings)} rebel embeddings")
    rdf2vec_index = faiss.read_index(rdf2vec_embeddings_file)
    rdf2vec_embeddings = [rdf2vec_index.reconstruct(i) for i in range(rdf2vec_index.ntotal)]
    print(f"read {len(rdf2vec_embeddings)} rdf2vec embeddings")

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(rebel_embeddings, rdf2vec_embeddings, test_size=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = REBELtoRDF2VecMapper().to(device)
    # # model = ResidualMLP().to(device)
    #
    # X_train_tensor = torch.tensor(X_train).to(device)
    # Y_train_tensor = torch.tensor(Y_train).to(device)
    # X_test_tensor = torch.tensor(X_test).to(device)
    # Y_test_tensor = torch.tensor(Y_test).to(device)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss()
    # loss_fn = cosine_loss
    #
    # for epoch in range(100):
    #     model.train()
    #     pred = model(X_train_tensor)
    #     loss = loss_fn(pred, Y_train_tensor)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     if (epoch + 1) % 5 == 0:
    #         model.eval()
    #         test_pred = model(X_test_tensor)
    #         test_loss = loss_fn(test_pred, Y_test_tensor)
    #         print(f"Epoch {epoch + 1} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
    #
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'input_dim': 1024,
    #     'output_dim': 100,
    # }, "rebel_to_rdf2vec_embedding_alignment_deep_MLP.pt")


    # checkpoint = torch.load("rebel_to_rdf2vec_embedding_alignment_residual_MLP.pt")
    # model = ResidualMLP(
    #     input_dim=checkpoint['input_dim'],
    #     output_dim=checkpoint['output_dim']
    # ).to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    #
    # visualize_spaces(rdf2vec_embeddings, rebel_embeddings, model, sample_size=300, title="Embedding Spaces (t-SNE) - Residual MLP")

    checkpoint = torch.load("rebel_to_rdf2vec_embedding_alignment_deep_MLP.pt")
    model = REBELtoRDF2VecMapper(
        input_dim=checkpoint['input_dim'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    visualize_spaces(rdf2vec_embeddings, rebel_embeddings, model, sample_size=300, title="Embedding Spaces (t-SNE) - Deep MLP")




if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("usage : rq2-embedding-alignment.py <rebel_embeddings_file> <rdf2vec_embeddings_file>")
    align_embeddings(sys.argv[1], sys.argv[2])
