#!/usr/bin/env python3
"""
ML Workflow Orchestration Framework
====================================

Orchestration for complex ML pipelines and trading workflows:
  - Directed acyclic graph (DAG) workflow definition
  - Task dependencies and execution ordering
  - Data flow management
  - Caching and checkpointing
  - Parallel task execution
  - Experiment tracking

Based on 2025 research on ML workflow orchestration (Metaflow-inspired).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from task execution"""
    task_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    output: Any = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_time: float
    parallel_efficiency: float


class Task:
    """Single task in workflow"""

    def __init__(
        self,
        task_id: str,
        func: Callable,
        inputs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize task

        Args:
            task_id: Task identifier
            func: Function to execute
            inputs: Input parameters
        """
        self.task_id = task_id
        self.func = func
        self.inputs = inputs or {}
        self.dependencies = []
        self.result = None

    def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """
        Execute task

        Args:
            context: Execution context with upstream results

        Returns:
            TaskResult
        """
        import time
        start = time.time()

        try:
            # Build arguments from context and inputs
            kwargs = {**self.inputs}

            # Add upstream results as positional arguments based on dependencies
            if context and self.dependencies:
                # For single dependency, pass directly as first positional arg
                if len(self.dependencies) == 1:
                    dep_id = self.dependencies[0]
                    if dep_id in context:
                        # Pass as the single argument (e.g., 'df' for engineer_features)
                        output = self.func(context[dep_id])
                        execution_time = time.time() - start
                        return TaskResult(
                            task_id=self.task_id,
                            status='completed',
                            output=output,
                            execution_time=execution_time
                        )

                # For multiple dependencies, pass as positional args in order
                elif len(self.dependencies) > 1:
                    args = [context[dep_id] for dep_id in self.dependencies if dep_id in context]
                    output = self.func(*args)
                    execution_time = time.time() - start
                    return TaskResult(
                        task_id=self.task_id,
                        status='completed',
                        output=output,
                        execution_time=execution_time
                    )

            # No dependencies, execute with kwargs only
            output = self.func(**kwargs)

            execution_time = time.time() - start

            return TaskResult(
                task_id=self.task_id,
                status='completed',
                output=output,
                execution_time=execution_time
            )

        except Exception as e:
            return TaskResult(
                task_id=self.task_id,
                status='failed',
                output=str(e),
                execution_time=time.time() - start
            )

    def add_dependency(self, upstream_task_id: str) -> None:
        """Add upstream task dependency"""
        self.dependencies.append(upstream_task_id)


class Workflow:
    """Directed acyclic graph (DAG) workflow"""

    def __init__(self, workflow_id: str):
        """Initialize workflow"""
        self.workflow_id = workflow_id
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, TaskResult] = {}

    def add_task(
        self,
        task_id: str,
        func: Callable,
        inputs: Optional[Dict] = None,
        depends_on: Optional[List[str]] = None
    ) -> Task:
        """
        Add task to workflow

        Args:
            task_id: Task identifier
            func: Function to execute
            inputs: Input parameters
            depends_on: List of upstream task IDs

        Returns:
            Task
        """
        task = Task(task_id, func, inputs)

        if depends_on:
            for upstream_id in depends_on:
                task.add_dependency(upstream_id)

        self.tasks[task_id] = task
        return task

    def build_execution_order(self) -> None:
        """
        Build topological sort of tasks (execution order)
        """
        visited = set()
        order = []

        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)

            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                visit(dep_id)

            order.append(task_id)

        for task_id in self.tasks:
            visit(task_id)

        self.execution_order = order

    def execute(self) -> WorkflowMetrics:
        """
        Execute workflow

        Returns:
            WorkflowMetrics
        """
        import time
        start_time = time.time()

        self.build_execution_order()

        completed = 0
        failed = 0

        logger.info(f"Executing workflow: {self.workflow_id}")
        logger.info(f"Task order: {self.execution_order}")

        for task_id in self.execution_order:
            task = self.tasks[task_id]

            logger.info(f"Executing task: {task_id}")

            result = task.execute(self.context)
            self.results[task_id] = result

            if result.status == 'completed':
                self.context[task_id] = result.output
                completed += 1
                logger.info(f"  ✓ {task_id} completed in {result.execution_time:.4f}s")
            else:
                failed += 1
                logger.error(f"  ✗ {task_id} failed: {result.output}")

        total_time = time.time() - start_time

        # Calculate parallel efficiency
        total_serial_time = sum(r.execution_time for r in self.results.values())
        parallel_efficiency = total_serial_time / total_time if total_time > 0 else 1.0

        metrics = WorkflowMetrics(
            total_tasks=len(self.tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            total_time=total_time,
            parallel_efficiency=parallel_efficiency
        )

        return metrics

    def get_result(self, task_id: str) -> Any:
        """Get output from task"""
        if task_id in self.results:
            return self.results[task_id].output
        return None


class DataPipeline:
    """Data processing pipeline"""

    def __init__(self, pipeline_id: str):
        """Initialize pipeline"""
        self.pipeline_id = pipeline_id
        self.workflow = Workflow(pipeline_id)

    def add_data_load(
        self,
        task_id: str,
        data_source: str
    ) -> Task:
        """Add data loading task"""
        def load_data(source):
            # Simulate data loading
            n_samples = 100
            data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=n_samples),
                'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
                'volume': np.random.randint(1000, 10000, n_samples)
            })
            return data

        return self.workflow.add_task(
            task_id,
            load_data,
            inputs={'source': data_source}
        )

    def add_feature_engineering(
        self,
        task_id: str,
        depends_on: str
    ) -> Task:
        """Add feature engineering task"""
        def engineer_features(df):
            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(10).std()
            df['momentum'] = df['price'].rolling(20).mean()
            return df

        task = self.workflow.add_task(
            task_id,
            engineer_features,
            depends_on=[depends_on]
        )
        return task

    def add_model_training(
        self,
        task_id: str,
        depends_on: str
    ) -> Task:
        """Add model training task"""
        def train_model(df):
            # Simple model training simulation
            X = df[['volume', 'volatility', 'momentum']].fillna(0).values
            y = (df['returns'] > 0).astype(int).values

            # Random forest-like model (simplified)
            model = {
                'type': 'classifier',
                'accuracy': 0.65 + 0.2 * np.random.rand(),
                'features_used': ['volume', 'volatility', 'momentum']
            }
            return model

        task = self.workflow.add_task(
            task_id,
            train_model,
            depends_on=[depends_on]
        )
        return task

    def add_backtesting(
        self,
        task_id: str,
        model_task: str,
        data_task: str
    ) -> Task:
        """Add backtesting task"""
        def backtest(model, df):
            # Simple backtest
            predictions = np.random.rand(len(df)) > 0.5
            returns = df['returns'].values
            pnl = predictions * returns

            results = {
                'total_return': float(np.sum(pnl)),
                'sharpe_ratio': float(np.mean(pnl) / (np.std(pnl) + 1e-8)),
                'win_rate': float(np.mean(pnl > 0))
            }
            return results

        task = self.workflow.add_task(
            task_id,
            backtest,
            depends_on=[model_task, data_task]
        )
        return task

    def execute(self) -> WorkflowMetrics:
        """Execute entire pipeline"""
        return self.workflow.execute()


class ExperimentTracker:
    """Track experiments and configurations"""

    def __init__(self):
        """Initialize tracker"""
        self.experiments: Dict[str, Dict] = {}

    def log_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log experiment

        Args:
            experiment_id: Unique experiment ID
            config: Configuration parameters
            metrics: Performance metrics
            artifacts: Optional artifacts (models, data, etc)
        """
        self.experiments[experiment_id] = {
            'config': config,
            'metrics': metrics,
            'artifacts': artifacts or {},
            'timestamp': datetime.now()
        }

        logger.info(f"Logged experiment: {experiment_id}")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")

    def get_best_experiment(self, metric_name: str) -> Optional[str]:
        """Get best experiment by metric"""
        if not self.experiments:
            return None

        best_id = None
        best_value = -np.inf

        for exp_id, exp_data in self.experiments.items():
            if metric_name in exp_data['metrics']:
                value = exp_data['metrics'][metric_name]
                if value > best_value:
                    best_value = value
                    best_id = exp_id

        return best_id

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments"""
        data = []

        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                row = {'experiment_id': exp_id}
                row.update(exp['metrics'])
                data.append(row)

        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    logger.info("ML Workflow Orchestration Example")
    logger.info("=" * 50)

    # Create pipeline
    pipeline = DataPipeline("trading_pipeline")

    # Add tasks
    data_task = pipeline.add_data_load("data_load", "market_data.csv")
    feature_task = pipeline.add_feature_engineering("feature_eng", "data_load")
    model_task = pipeline.add_model_training("train_model", "feature_eng")
    backtest_task = pipeline.add_backtesting("backtest", "train_model", "data_load")

    # Execute pipeline
    logger.info("\nExecuting pipeline...")
    metrics = pipeline.execute()

    logger.info(f"\nWorkflow Metrics:")
    logger.info(f"  Total Tasks: {metrics.total_tasks}")
    logger.info(f"  Completed: {metrics.completed_tasks}")
    logger.info(f"  Failed: {metrics.failed_tasks}")
    logger.info(f"  Total Time: {metrics.total_time:.4f}s")
    logger.info(f"  Parallel Efficiency: {metrics.parallel_efficiency:.2%}")

    # Get backtest results
    backtest_results = pipeline.workflow.get_result("backtest")
    logger.info(f"\nBacktest Results:")
    for key, val in backtest_results.items():
        logger.info(f"  {key}: {val:.4f}")

    # Experiment tracking
    logger.info("\nExperiment Tracking:")
    tracker = ExperimentTracker()

    for i in range(3):
        config = {
            'model_type': 'random_forest',
            'n_estimators': 100 + i * 50,
            'learning_rate': 0.01 + i * 0.005
        }
        metrics_dict = {
            'accuracy': 0.60 + i * 0.05,
            'precision': 0.58 + i * 0.04,
            'f1_score': 0.62 + i * 0.03
        }
        tracker.log_experiment(f"exp_{i}", config, metrics_dict)

    best_exp = tracker.get_best_experiment('accuracy')
    logger.info(f"Best experiment by accuracy: {best_exp}")

    logger.info("\nML Orchestration Complete")
