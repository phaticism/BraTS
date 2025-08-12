import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
import settings


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of metrics."""
    # Calculate the total number of examples used during training
    total_examples = sum([num_examples for num_examples, _ in metrics])

    # Create a dictionary of metric name to weighted average value
    weighted_metrics = {}
    for metric_name in metrics[0][1].keys():
        weighted_metrics[metric_name] = sum(
            [num_examples * m[metric_name] for num_examples, m in metrics]
        ) / total_examples

    return weighted_metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """
    # Define the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=3,  # Minimum number of clients for training
        min_evaluate_clients=2,  # Minimum number of clients for evaluation
        min_available_clients=3,  # Minimum number of total clients needed
        # Use weighted average for metrics
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda _: {
            "epochs": settings.EPOCHS,
            "batch_size": settings.BATCH_SIZE,
        },
        on_evaluate_config_fn=lambda _: {
            "batch_size": settings.BATCH_SIZE,
        },
    )

    config = ServerConfig(num_rounds=10)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)
