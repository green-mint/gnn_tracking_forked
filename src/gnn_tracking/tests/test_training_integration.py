from __future__ import annotations

from functools import partial

import pytest
from torch_geometric.loader import DataLoader

from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightBCELoss,
    LossFctType,
    PotentialLoss,
)
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.dynamiclossweights import NormalizeAt
from gnn_tracking.training.tcn_trainer import TCNTrainer, TrainingTruthCutConfig
from gnn_tracking.utils.seeds import fix_seeds


@pytest.mark.parametrize("loss_weights", [("default"), ("auto")])
def test_train(tmp_path, built_graphs, loss_weights: str) -> None:
    fix_seeds()
    _, graph_builder = built_graphs
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]

    graphs = graph_builder.data_list
    n_graphs = len(graphs)
    assert n_graphs == 2
    params = {"batch_size": 1, "num_workers": 1}

    loaders = {
        "train": DataLoader([graphs[0]], **params, shuffle=True),
        "test": DataLoader([graphs[1]], **params),
        "val": DataLoader([graphs[1]], **params),
    }

    q_min, sb = 0.01, 0.1
    loss_functions: dict[str, LossFctType] = {
        "edge": EdgeWeightBCELoss(),
        "potential": PotentialLoss(q_min=q_min),
        "background": BackgroundLoss(sb=sb),
    }

    # set up a model and trainer
    model = GraphTCN(
        node_indim,
        edge_indim,
        h_dim=2,
        hidden_dim=64,
        interaction_edge_hidden_dim=3,
        interaction_node_hidden_dim=3,
    )

    _loss_weights = None
    if loss_weights == "default":
        _loss_weights = None
    elif loss_weights == "auto":
        _loss_weights = NormalizeAt(at=[0])
    else:
        raise ValueError()

    trainer = TCNTrainer(
        model=model,
        loaders=loaders,
        loss_functions=loss_functions,
        lr=0.0001,
        cluster_functions={"dbscan": partial(dbscan_scan, n_trials=1)},  # type: ignore
        loss_weights=_loss_weights,
    )
    trainer.checkpoint_dir = tmp_path
    tcc = TrainingTruthCutConfig(
        pt_thld=0.01,
        without_noise=True,
        without_non_reconstructable=True,
    )
    trainer.training_truth_cuts = tcc

    trainer.train(epochs=2, max_batches=1)
    trainer.test_step()
    trainer.save_checkpoint("model.pt")
    trainer.load_checkpoint("model.pt")
