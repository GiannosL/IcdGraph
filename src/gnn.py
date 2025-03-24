import tqdm
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from src import EdgeType
from src.nn_stuff.results import ResultsGNN
from src.nn_stuff.early_stop import EarlyStop
from src.nn_stuff.link_pred import LinkPredictor


def run_gnn(
        graph: HeteroData,
        edge_type: EdgeType,
        epochs: int = 5,
        itta: float = 0.01,
        classification_type: str = 'linear',
        ) -> ResultsGNN:
    # get running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}\n')

    # data split
    transform = RandomLinkSplit(
        num_test=0.1,
        is_undirected=True,
        neg_sampling_ratio=2.0,
        edge_types=[edge_type()]
    )
    train_data, val_data, test_data = transform(graph)

    # initialize link predictor
    model = LinkPredictor(
        edge_type=edge_type,
        train_graph=train_data,
        classifier_type=classification_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=itta)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stop = EarlyStop()

    # place data to device
    train_data.to(device=device)
    test_data.to(device=device)
    val_data.to(device=device)
    model.to(device=device)

    # init emb spaces
    emb_space_train, emb_space_val, emb_space_test = None, None, None

    loss_history, val_loss_history = [], []
    for _ in tqdm.tqdm(range(epochs), desc='Epoch'):
        model.train()
        optimizer.zero_grad()

        y_hat, emb_space_train = model(graph=train_data)
        loss = criterion(y_hat, train_data[edge_type()].edge_label)
        loss_history.append(loss.item())
        early_stop(metric=loss.item())

        loss.backward()
        optimizer.step()

        # model validation
        with torch.no_grad():
            model.eval()
            validation_y, emb_space_val = model(graph=val_data)
            validation_loss = criterion(validation_y, val_data[edge_type()].edge_label)
            val_loss_history.append(validation_loss.item())

        if early_stop.early_stop:
            break

    # test-set results
    with torch.no_grad():
        model.eval()
        y_test, emb_space_test = model(graph=test_data)

    # results object
    results = ResultsGNN(edge_type=edge_type)
    results.set_test_results(r=y_test,
                             edge_label_index=test_data[edge_type()].edge_label_index,
                             edge_label=test_data[edge_type()].edge_label)
    results.set_val_results(r=validation_y,
                            edge_label_index=val_data[edge_type()].edge_label_index,
                            edge_label=val_data[edge_type()].edge_label)
    results.set_training_results(r=y_hat,
                                 edge_label_index=train_data[edge_type()].edge_label_index,
                                 edge_label=train_data[edge_type()].edge_label)
    results.set_loss_history(history=loss_history)
    results.set_training_embeds(embs=emb_space_train)
    results.set_validation_embeds(embs=emb_space_val)
    results.set_test_embeds(embs=emb_space_test)
    results.set_validation_loss_history(history=val_loss_history)

    return results
