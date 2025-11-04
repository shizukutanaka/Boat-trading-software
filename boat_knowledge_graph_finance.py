#!/usr/bin/env python3
"""
Knowledge Graph for Financial Networks
=======================================

Entity-relationship graphs for financial data:
  - Company/entity relationship modeling
  - Event-enhanced knowledge graph construction
  - Systemic risk detection via graph analysis
  - Portfolio construction from KG
  - Fraud detection through network analysis

Based on 2025 research (FinDKG, FinKario, knowledge graphs in finance).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Knowledge graph entity (company, index, indicator)"""
    entity_id: str
    entity_type: str  # 'company', 'index', 'economic_indicator'
    attributes: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Relation:
    """Relationship between entities"""
    source_id: str
    target_id: str
    relation_type: str  # 'owns', 'competes_with', 'supplies', 'correlated'
    strength: float  # 0 to 1
    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphMetrics:
    """Financial network metrics"""
    density: float
    clustering_coefficient: float
    systemic_importance: Dict[str, float]  # Risk scores per entity
    community_structure: List[Set[str]]  # Connected components


class KnowledgeGraph:
    """Financial knowledge graph"""

    def __init__(self, name: str = "FinancialKG"):
        """Initialize knowledge graph"""
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, float] = None) -> Entity:
        """Add entity to graph"""
        entity = Entity(entity_id=entity_id, entity_type=entity_type, attributes=attributes or {})
        self.entities[entity_id] = entity
        return entity

    def add_relation(self, source_id: str, target_id: str, relation_type: str, strength: float = 1.0):
        """Add relation between entities"""
        relation = Relation(
            source_id=source_id, target_id=target_id, relation_type=relation_type, strength=strength
        )
        self.relations.append(relation)
        self.adjacency[source_id].add(target_id)

    def get_neighbors(self, entity_id: str) -> Set[str]:
        """Get neighboring entities"""
        return self.adjacency.get(entity_id, set())

    def compute_entity_embedding(self, entity_id: str, embedding_dim: int = 32) -> np.ndarray:
        """
        Compute entity embedding via random walk

        Args:
            entity_id: Entity to embed
            embedding_dim: Embedding dimension

        Returns:
            Entity embedding
        """
        # Simple embedding: one-hot over neighbors
        neighbors = self.get_neighbors(entity_id)
        embedding = np.zeros(embedding_dim)

        for i, neighbor_id in enumerate(list(neighbors)[: embedding_dim]):
            embedding[i] = 1.0 / (len(neighbors) + 1)

        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def detect_cycles(self, max_depth: int = 3) -> List[List[str]]:
        """Detect cycles in the knowledge graph (feedback loops)"""
        cycles = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if current in visited and current in path:
                # Found cycle
                cycle_start = path.index(current)
                cycles.append(path[cycle_start:] + [current])
                return

            if len(path) > max_depth:
                return

            visited.add(current)
            path.append(current)

            for neighbor in self.get_neighbors(current):
                dfs(neighbor, path.copy(), visited.copy())

        for entity_id in self.entities:
            dfs(entity_id, [], set())

        return cycles

    def compute_pagerank(self, damping_factor: float = 0.85, iterations: int = 10) -> Dict[str, float]:
        """
        Compute PageRank for systemic importance

        Args:
            damping_factor: Damping factor
            iterations: Iteration count

        Returns:
            Entity importance scores
        """
        n_entities = len(self.entities)
        entity_ids = list(self.entities.keys())
        entity_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # Initialize scores
        scores = {eid: 1.0 / n_entities for eid in entity_ids}

        for _ in range(iterations):
            new_scores = {}
            for entity_id in entity_ids:
                # Incoming edges
                incoming = [r for r in self.relations if r.target_id == entity_id]
                rank_sum = sum(scores[r.source_id] / (len(self.adjacency.get(r.source_id, set())) + 1)
                              for r in incoming)

                new_scores[entity_id] = (1 - damping_factor) / n_entities + damping_factor * rank_sum

            scores = new_scores

        return scores

    def compute_centrality(self) -> Dict[str, float]:
        """Compute betweenness centrality (brokerage importance)"""
        centrality = {eid: 0.0 for eid in self.entities}

        # Simplified: degree-based
        for eid in self.entities:
            in_degree = sum(1 for r in self.relations if r.target_id == eid)
            out_degree = len(self.adjacency.get(eid, set()))
            centrality[eid] = (in_degree + out_degree) / (2 * len(self.relations) + 1)

        return centrality


class EventEnhancedKG:
    """Knowledge graph enhanced with events"""

    def __init__(self, base_kg: KnowledgeGraph):
        """Initialize event-enhanced KG"""
        self.base_kg = base_kg
        self.events: List[Dict] = []

    def add_event(self, event_type: str, affected_entities: List[str], impact: float, timestamp: int = 0):
        """
        Add event to KG

        Args:
            event_type: 'earnings_miss', 'merger', 'regulation', etc.
            affected_entities: Entities affected by event
            impact: Impact magnitude (-1 to +1)
            timestamp: Event time
        """
        event = {
            "type": event_type,
            "affected_entities": affected_entities,
            "impact": impact,
            "timestamp": timestamp,
        }
        self.events.append(event)

        # Update entity attributes
        for entity_id in affected_entities:
            if entity_id in self.base_kg.entities:
                self.base_kg.entities[entity_id].attributes[f"event_{event_type}"] = impact

    def propagate_impact(self) -> Dict[str, float]:
        """
        Propagate event impacts through network

        Returns:
            Total impact per entity
        """
        impact_scores = {eid: 0.0 for eid in self.base_kg.entities}

        for event in self.events:
            affected = set(event["affected_entities"])
            impact = event["impact"]

            # BFS propagation
            frontier = affected.copy()
            visited = affected.copy()
            current_impact = impact

            while frontier and abs(current_impact) > 0.01:
                next_frontier = set()
                for entity_id in frontier:
                    impact_scores[entity_id] += current_impact
                    # Spread to neighbors with decay
                    for neighbor_id in self.base_kg.get_neighbors(entity_id):
                        if neighbor_id not in visited:
                            next_frontier.add(neighbor_id)
                            visited.add(neighbor_id)

                frontier = next_frontier
                current_impact *= 0.7  # Decay factor

        return impact_scores


class FraudDetectionKG:
    """Fraud detection via KG analysis"""

    @staticmethod
    def detect_anomalous_patterns(kg: KnowledgeGraph) -> List[Tuple[str, float]]:
        """
        Detect anomalous patterns in financial network

        Args:
            kg: Knowledge graph

        Returns:
            List of (entity_id, anomaly_score)
        """
        anomaly_scores = []

        # Pattern 1: High clustering (potential fraud ring)
        pagerank = kg.compute_pagerank()

        # Pattern 2: Unusual connectivity
        centrality = kg.compute_centrality()

        for entity_id in kg.entities:
            # Anomaly if high importance but few relations
            importance = pagerank[entity_id]
            connectivity = centrality[entity_id]

            # Anomalous if disproportionate
            anomaly = abs(importance - connectivity)
            anomaly_scores.append((entity_id, float(anomaly)))

        return sorted(anomaly_scores, key=lambda x: x[1], reverse=True)


class PortfolioConstructionViaKG:
    """Portfolio construction from knowledge graph"""

    @staticmethod
    def construct_portfolio(kg: KnowledgeGraph, budget: float = 100.0) -> Dict[str, float]:
        """
        Construct portfolio based on graph centrality

        Args:
            kg: Knowledge graph of companies
            budget: Total budget

        Returns:
            Portfolio weights
        """
        # Get importance scores
        pagerank = kg.compute_pagerank()

        # Filter to company entities
        companies = {eid: entity for eid, entity in kg.entities.items() if entity.entity_type == "company"}

        # Weights proportional to importance
        importance_scores = {eid: pagerank.get(eid, 0.0) for eid in companies}

        total_score = sum(importance_scores.values())

        if total_score == 0:
            # Equal weight
            n_companies = len(companies)
            return {eid: budget / n_companies for eid in companies}

        portfolio = {eid: (score / total_score) * budget for eid, score in importance_scores.items()}

        return portfolio


if __name__ == "__main__":
    logger.info("Knowledge Graph for Financial Networks")
    logger.info("=" * 50)

    # Build financial knowledge graph
    kg = KnowledgeGraph(name="StockMarketKG")

    # Add companies
    companies = ["TECH_A", "TECH_B", "BANK_X", "RETAIL_Y"]
    for company in companies:
        kg.add_entity(company, "company", {"market_cap": np.random.uniform(1e9, 1e11)})

    # Add indices
    kg.add_entity("TECH_INDEX", "index", {"value": 100})
    kg.add_entity("BANK_INDEX", "index", {"value": 100})

    logger.info("\nAdding Relations (ownership, competition, supply chains)")
    # Relations
    kg.add_relation("TECH_A", "TECH_B", "competes_with", 0.8)
    kg.add_relation("TECH_A", "BANK_X", "borrows_from", 0.6)
    kg.add_relation("TECH_B", "RETAIL_Y", "supplies", 0.7)
    kg.add_relation("BANK_X", "RETAIL_Y", "finances", 0.9)

    logger.info(f"Graph has {len(kg.entities)} entities and {len(kg.relations)} relations")

    # Compute metrics
    logger.info("\nComputing Graph Metrics")
    pagerank = kg.compute_pagerank()
    logger.info("  PageRank (Systemic Importance):")
    for eid, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:4]:
        logger.info(f"    {eid}: {score:.4f}")

    centrality = kg.compute_centrality()
    logger.info("  Centrality (Brokerage):")
    for eid in companies[:2]:
        logger.info(f"    {eid}: {centrality[eid]:.4f}")

    # Event-enhanced KG
    logger.info("\nAdding Events to Knowledge Graph")
    ekg = EventEnhancedKG(kg)
    ekg.add_event("earnings_miss", ["TECH_A"], -0.3, timestamp=0)
    ekg.add_event("merger", ["TECH_B", "BANK_X"], 0.2, timestamp=1)

    impact = ekg.propagate_impact()
    logger.info("  Event Impact Propagation:")
    for eid in companies[:3]:
        logger.info(f"    {eid}: {impact[eid]:.4f}")

    # Fraud detection
    logger.info("\nFraud Detection via KG Analysis")
    anomalies = FraudDetectionKG.detect_anomalous_patterns(kg)
    logger.info("  Top Anomalies:")
    for entity_id, score in anomalies[:3]:
        logger.info(f"    {entity_id}: {score:.4f}")

    # Portfolio construction
    logger.info("\nPortfolio Construction via KG")
    portfolio = PortfolioConstructionViaKG.construct_portfolio(kg, budget=100.0)
    logger.info("  Portfolio Weights (budget=100):")
    for company, weight in portfolio.items():
        logger.info(f"    {company}: {weight:.2f}")

    logger.info("\nKnowledge Graph Analysis Complete")
