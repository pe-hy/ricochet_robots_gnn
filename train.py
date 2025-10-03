import os
import pickle
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

class RicochetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, indices: list):
        """
        Args:
            data_path: Path to the pickle file containing all graphs
            indices: List of graph indices to include in this split
        """
        with open(data_path, 'rb') as f:
            all_graphs = pickle.load(f)
        
        self.data = [all_graphs[i] for i in indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class GNNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float, num_classes: int = 1, heads: int = 4):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection layer
        self.input_conv = GATv2Conv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=4,
            add_self_loops=False
        )
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Single recurrent hidden layer (reused N-2 times)
        self.hidden_conv = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=4,
            add_self_loops=False
        )

        # Output layer (single head)
        self.output_conv = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=4,
            add_self_loops=False
        )

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # First layer
        identity = self.input_proj(x)
        x = self.input_conv(x, edge_index, edge_attr=edge_attr)
        x = x + identity
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Recurrent hidden layers (apply same layer N-2 times)
        for i in range(self.num_layers - 2):
            identity = x
            x = self.hidden_conv(x, edge_index, edge_attr=edge_attr)
            x = x + identity
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer
        identity = x
        x = self.output_conv(x, edge_index, edge_attr=edge_attr)
        x = x + identity

        return self.classifier(x)


class RicochetGNNModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, test_loader=None):
        super().__init__()
        self.save_hyperparameters(ignore=['test_loader'])
        self.cfg = cfg
        self.test_loader_stored = test_loader
        self.task = cfg.data.task
        
        # Determine number of output classes based on task
        if self.task == 'binning':
            num_classes = cfg.model.get('num_bins', 100)  # Default to 100 bins if not specified
        else:
            num_classes = 1
        
        self.model = GNNModel(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=num_classes,
            heads=cfg.model.get('heads', 4)
        )
        
        self.val_preds = []
        self.val_labels = []
        
        self.test_preds = []
        self.test_labels = []
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.model(x, edge_index, edge_attr, batch)
    
    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        logits = self(x, edge_index, edge_attr, batch.batch)

        # Select loss function based on task
        if self.task == 'regression':
            # MSE loss for regression (excluding -1 labels)
            y = batch.y.unsqueeze(-1)
            mask = batch.y != -1
            if mask.sum() > 0:
                loss = F.mse_loss(logits[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        elif self.task == 'binning':
            # Cross entropy for multi-class classification
            y = batch.y
            # Filter out -1 labels (non-main components)
            mask = y != -1
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            # Binary cross entropy for binary tasks
            y = batch.y.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        logits = self(x, edge_index, edge_attr, batch.batch)

        # Select loss function based on task
        if self.task == 'regression':
            y = batch.y.unsqueeze(-1)
            mask = batch.y != -1
            if mask.sum() > 0:
                loss = F.mse_loss(logits[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=logits.device)
            preds = logits  # Use raw predictions for regression
        elif self.task == 'binning':
            y = batch.y
            mask = y != -1
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=logits.device)
            preds = torch.argmax(logits, dim=-1)  # Get predicted class
        else:
            y = batch.y.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()

        # Store predictions and labels per graph
        for graph_id in range(batch.num_graphs):
            mask = batch.batch == graph_id
            self.val_preds.append(preds[mask].cpu())
            self.val_labels.append(batch.y[mask].cpu())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        logits = self(x, edge_index, edge_attr, batch.batch)

        # Select loss function based on task
        if self.task == 'regression':
            y = batch.y.unsqueeze(-1)
            mask = batch.y != -1
            if mask.sum() > 0:
                loss = F.mse_loss(logits[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=logits.device)
            preds = logits
        elif self.task == 'binning':
            y = batch.y
            mask = y != -1
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=logits.device)
            preds = torch.argmax(logits, dim=-1)
        else:
            y = batch.y.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()

        # Store predictions and labels per graph
        for graph_id in range(batch.num_graphs):
            mask = batch.batch == graph_id
            self.test_preds.append(preds[mask].cpu())
            self.test_labels.append(batch.y[mask].cpu())

        self.log('test_loss', loss, batch_size=batch.num_graphs)
        return loss
    
    def on_validation_epoch_end(self):
        self._compute_metrics(self.val_preds, self.val_labels, 'val')
        self.val_preds.clear()
        self.val_labels.clear()

        # Run test evaluation every 5 epochs
        if (self.current_epoch + 1) % 5 == 0 and self.test_loader_stored is not None:
            print(f"\n{'='*60}")
            print(f"Running test evaluation at epoch {self.current_epoch + 1}")
            print(f"{'='*60}")

            # Run test evaluation
            test_preds = []
            test_labels = []
            test_graphs_with_preds = [] if self.task == 'best_component' else None

            self.eval()
            with torch.no_grad():
                for batch in self.test_loader_stored:
                    batch = batch.to(self.device)
                    x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

                    logits = self(x, edge_index, edge_attr, batch.batch)

                    # Get predictions based on task
                    if self.task == 'regression':
                        preds = logits
                    elif self.task == 'binning':
                        preds = torch.argmax(logits, dim=-1)
                    else:
                        preds = (torch.sigmoid(logits) > 0.5).float()

                    # Store per-graph predictions
                    for graph_id in range(batch.num_graphs):
                        mask = batch.batch == graph_id
                        graph_preds = preds[mask].cpu()
                        graph_labels = batch.y[mask].cpu()

                        test_preds.append(graph_preds)
                        test_labels.append(graph_labels)

                        # Save full graph data only for best_component task
                        if self.task == 'best_component':
                            # Get edge mask for this graph
                            edge_mask = (batch.batch[batch.edge_index[0]] == graph_id) & \
                                       (batch.batch[batch.edge_index[1]] == graph_id)

                            # Create new Data object with original data + predictions as y2
                            graph_data = Data(
                                x=batch.x[mask].cpu(),
                                edge_index=batch.edge_index[:, edge_mask].cpu(),
                                edge_attr=batch.edge_attr[edge_mask].cpu(),
                                y=graph_labels,
                                y2=graph_preds.squeeze()
                            )

                            # Renumber edge_index to be relative to this graph (0-indexed)
                            if graph_data.edge_index.numel() > 0:
                                node_offset = mask.nonzero(as_tuple=True)[0][0].to(graph_data.edge_index.device)
                                graph_data.edge_index = graph_data.edge_index - node_offset

                            test_graphs_with_preds.append(graph_data)

            self.train()

            # Compute and log test metrics (returns exact match ratio)
            exact_match_ratio = self._compute_metrics(test_preds, test_labels, 'test', log_to_wandb=True)

            # Save predictions to pickle file (only for best_component task)
            if self.task == 'best_component':
                save_dir = Path('test_predictions')
                save_dir.mkdir(exist_ok=True)

                filename = f"test_predictions_{self.current_epoch + 1}_{exact_match_ratio:.4f}.pkl"
                save_path = save_dir / filename

                with open(save_path, 'wb') as f:
                    pickle.dump(test_graphs_with_preds, f)

                print(f"Saved test predictions to {save_path}")
    
    def on_test_epoch_end(self):
        self._compute_metrics(self.test_preds, self.test_labels, 'test_final')
        self.test_preds.clear()
        self.test_labels.clear()
    
    def _compute_metrics(self, preds_list, labels_list, prefix, log_to_wandb=True):
        """
        Compute metrics where preds_list and labels_list are lists of per-graph predictions.
        Each element in the list corresponds to one graph.
        Returns exact_match_ratio.
        """
        # Flatten all predictions and labels for node-level metrics
        all_preds = torch.cat(preds_list).squeeze().numpy()
        all_labels = torch.cat(labels_list).squeeze().numpy()

        # Compute metrics based on task type
        if self.task == 'regression':
            # For regression, use MAE and RMSE (excluding -1 labels)
            mask = all_labels != -1
            mae = np.mean(np.abs(all_preds[mask] - all_labels[mask]))
            rmse = np.sqrt(np.mean((all_preds[mask] - all_labels[mask]) ** 2))
        else:
            # For classification tasks
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0, average='binary' if self.task != 'binning' else 'macro')
            recall = recall_score(all_labels, all_preds, zero_division=0, average='binary' if self.task != 'binning' else 'macro')
            f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary' if self.task != 'binning' else 'macro')

        # Graph-level exact match
        num_graphs = len(preds_list)
        exact_matches = 0

        for graph_preds, graph_labels in zip(preds_list, labels_list):
            graph_preds = graph_preds.squeeze().numpy()
            graph_labels = graph_labels.squeeze().numpy()

            if self.task == 'regression':
                # For regression, consider exact if within tolerance
                if np.allclose(graph_preds, graph_labels, atol=1.0):
                    exact_matches += 1
            else:
                if (graph_preds == graph_labels).all():
                    exact_matches += 1

        exact_match_ratio = exact_matches / num_graphs
        
        # Log to wandb based on prefix
        if log_to_wandb:
            if prefix == 'val':
                # Log validation metrics
                if self.task == 'regression':
                    self.log('val_mae', mae, prog_bar=True)
                    self.log('val_rmse', rmse, prog_bar=False)
                else:
                    self.log('val_accuracy', accuracy, prog_bar=True)
                    self.log('val_precision', precision, prog_bar=False)
                    self.log('val_recall', recall, prog_bar=False)
                self.log('val_exact_match', exact_match_ratio, prog_bar=True)
            elif prefix == 'test':
                # Log only exact match for periodic test evaluation
                self.log('test_exact_match', exact_match_ratio, prog_bar=True)
            elif prefix == 'test_final':
                # Log final test metrics
                if self.task == 'regression':
                    self.log('test_final_mae', mae)
                    self.log('test_final_rmse', rmse)
                else:
                    self.log('test_final_accuracy', accuracy)
                    self.log('test_final_precision', precision)
                    self.log('test_final_recall', recall)
                self.log('test_final_exact_match', exact_match_ratio)
        
        # Print to console
        if self.task == 'regression':
            print(f"\n{prefix.upper()} Metrics:")
            print(f"  MAE:          {mae:.4f}")
            print(f"  RMSE:         {rmse:.4f}")
            print(f"  Exact Match (±1): {exact_match_ratio:.4f} ({exact_matches}/{num_graphs})")
        else:
            print(f"\n{prefix.upper()} Metrics:")
            print(f"  Accuracy:     {accuracy:.4f}")
            print(f"  Precision:    {precision:.4f}")
            print(f"  Recall:       {recall:.4f}")
            print(f"  F1 Score:     {f1:.4f}")
            print(f"  Exact Match:  {exact_match_ratio:.4f} ({exact_matches}/{num_graphs})")

        return exact_match_ratio
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.max_lr,
            weight_decay=self.cfg.training.weight_decay
        )
        
        # Cosine annealing with warmup
        warmup_epochs = self.cfg.training.warmup_epochs
        max_epochs = self.cfg.training.max_epochs
        
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                progress = (current_epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    pl.seed_everything(cfg.seed)
    
    # Task mapping to data files
    task_to_file = {
        'basic': 'train_data_basic_data.pkl',
        'threshold': 'train_data_threshold_data.pkl',
        'unreachable': 'train_data_unreachable_components_data.pkl',
        'regression': 'train_data_regression_data.pkl',
        'binning': 'train_data_binning_data.pkl',
        'best_component': 'train_data_best_component_data.pkl'
    }
    
    # Get task from config
    task = cfg.data.task
    if task not in task_to_file:
        raise ValueError(f"Invalid task: {task}. Must be one of {list(task_to_file.keys())}")
    
    # Load the appropriate data file
    data_file = task_to_file[task]
    data_path = Path(cfg.data.data_dir) / data_file
    print(f"\nTask: {task}")
    print(f"Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        all_graphs = pickle.load(f)
    
    total_graphs = len(all_graphs)
    print(f"Total graphs loaded: {total_graphs}")
    
    # Create splits
    val_size = cfg.data.val_size
    test_size = cfg.data.test_size
    
    all_indices = list(range(total_graphs))
    import random
    random.seed(cfg.seed)
    random.shuffle(all_indices)
    train_indices = all_indices[:-val_size - test_size]
    val_indices = all_indices[-val_size - test_size:-test_size]
    test_indices = all_indices[-test_size:]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_indices)} graphs")
    print(f"  Val:   {len(val_indices)} graphs")
    print(f"  Test:  {len(test_indices)} graphs")
    
    # Create datasets
    train_dataset = RicochetDataset(data_path, train_indices)
    val_dataset = RicochetDataset(data_path, val_indices)
    test_dataset = RicochetDataset(data_path, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    # Initialize model with test_loader for periodic evaluation
    model = RicochetGNNModule(cfg, test_loader=test_loader)
    
    # Setup logger with task-specific run name
    run_name = f"{cfg.wandb.run_name}-{task}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_exact_match',
        mode='max',
        save_top_k=3,
        filename=f'ricochet-{task}-{{epoch:02d}}-{{val_exact_match:.4f}}'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    trainer.test(model, test_loader, ckpt_path='best')
    
    wandb.finish()


if __name__ == "__main__":
    main()