# File: core/manifold_os.py
"""
Manifold OS: Universal Constructor with Driver Management

The OS implements the ICASRA Universal Constructor pattern:
- **A (Constructor)**: Validates and commits states
- **B (Copier)**: Reproduces states with structural preservation
- **C (Controller)**: Coordinates driver lifecycle and resources
- **D (Interface)**: Manages external data ingestion

Design Principles:
- Immutability: All data structures are immutable
- Idempotence: Operations can be repeated safely
- Content Addressability: Everything identified by content hash
- No Scheduling: Immutability eliminates need for complex scheduling
- Driver Lifecycle: Wake → Active → Idle → Remove (or Restart)

Driver Management:
- Drivers are ephemeral, stateless workers
- OS manages driver lifecycle (wake, idle, restart, remove)
- Processes cannot harm each other (pure functional)
- First driver: IngestDriver (tokenizes external data)
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Callable, Any, Tuple, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import time
import json
import hashlib
from pathlib import Path
import threading

from .kernel import Kernel, HLLSet, compute_sha1, Morphism, SingularityReport
from .hrt import (
    HRT, HRTConfig, AdjacencyMatrix, 
    HLLSetLattice, BasicHLLSet, Cover
)
from .entanglement import (
    EntanglementEngine, EntanglementMorphism,
    compute_hrt_entanglement, compute_entanglement_network
)
from .state_mapping import (
    StateMapping, StateMapper, HLLSetSwarm, TrajectoryTriple,
    map_mos_states, create_state_swarm
)


# =============================================================================
# SECTION 1: Driver Abstraction (Universal Constructor Pattern)
# =============================================================================

class DriverState(Enum):
    """Driver lifecycle states."""
    CREATED = "created"       # Just instantiated
    IDLE = "idle"             # Waiting for work
    ACTIVE = "active"         # Processing
    ERROR = "error"           # Failed, needs restart
    DEAD = "dead"             # Terminated, needs removal


@dataclass
class DriverStats:
    """Statistics for driver monitoring."""
    operations_count: int = 0
    errors_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: Optional[float] = None
    total_active_time: float = 0.0
    
    def record_operation(self, duration: float):
        """Record successful operation."""
        self.operations_count += 1
        self.last_active = time.time()
        self.total_active_time += duration
    
    def record_error(self):
        """Record error."""
        self.errors_count += 1


class Driver:
    """
    Base driver class.
    
    Drivers are stateless, ephemeral workers that perform specific operations.
    They follow the lifecycle: CREATED → IDLE → ACTIVE → (IDLE | ERROR | DEAD)
    """
    
    def __init__(self, driver_id: str, driver_type: str):
        self.driver_id = driver_id
        self.driver_type = driver_type
        self.state = DriverState.CREATED
        self.stats = DriverStats()
        self._lock = threading.Lock()
    
    def wake(self):
        """Wake driver from idle state."""
        with self._lock:
            if self.state in (DriverState.CREATED, DriverState.IDLE):
                self.state = DriverState.IDLE
                return True
            return False
    
    def activate(self):
        """Activate driver for processing."""
        with self._lock:
            if self.state == DriverState.IDLE:
                self.state = DriverState.ACTIVE
                return True
            return False
    
    def idle(self):
        """Put driver to idle state."""
        with self._lock:
            if self.state == DriverState.ACTIVE:
                self.state = DriverState.IDLE
                return True
            return False
    
    def mark_error(self):
        """Mark driver as errored."""
        with self._lock:
            self.state = DriverState.ERROR
            self.stats.record_error()
    
    def mark_dead(self):
        """Mark driver as dead (needs removal)."""
        with self._lock:
            self.state = DriverState.DEAD
    
    def restart(self):
        """Restart driver (error → idle)."""
        with self._lock:
            if self.state == DriverState.ERROR:
                self.state = DriverState.IDLE
                return True
            return False
    
    def process(self, *args, **kwargs) -> Any:
        """Override in subclass to implement processing logic."""
        raise NotImplementedError("Subclass must implement process()")
    
    @property
    def is_alive(self) -> bool:
        """Check if driver is alive (not dead)."""
        return self.state != DriverState.DEAD
    
    @property
    def needs_restart(self) -> bool:
        """Check if driver needs restart."""
        return self.state == DriverState.ERROR


# =============================================================================
# SECTION 2: Ingest Driver (D - Interface to External Reality)
# =============================================================================

@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""
    min_token_length: int = 1
    max_token_length: int = 100
    lowercase: bool = True
    remove_punctuation: bool = False
    split_on: str = " "


class IngestDriver(Driver):
    """
    Ingest Driver: The gateway between external reality and the system.
    
    Responsibilities:
    1. Ingest raw external data (text, events, observations)
    2. Tokenize data into meaningful units
    3. Convert tokens to HLLSets (via kernel)
    4. Return content-addressed HLLSet
    
    This implements the D (Interface) component of ICASRA.
    """
    
    def __init__(self, driver_id: str, config: Optional[TokenizationConfig] = None):
        super().__init__(driver_id, "ingest")
        self.config = config or TokenizationConfig()
        self.ingested_count = 0
    
    def tokenize(self, raw_data: str) -> Set[str]:
        """
        Tokenize raw data into tokens.
        
        Args:
            raw_data: Raw string data from external reality
        
        Returns:
            Set of tokens
        """
        if not raw_data:
            return set()
        
        # Apply transformations
        text = raw_data
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens
        tokens = text.split(self.config.split_on)
        
        # Filter by length
        tokens = [
            t.strip() for t in tokens 
            if self.config.min_token_length <= len(t.strip()) <= self.config.max_token_length
        ]
        
        return set(tokens)
    
    def process(self, raw_data: str, kernel: Kernel) -> HLLSet:
        """
        Process raw data through ingestion pipeline.
        
        Args:
            raw_data: Raw string data
            kernel: Kernel for HLLSet creation
        
        Returns:
            Content-addressed HLLSet
        """
        if not self.activate():
            raise RuntimeError(f"Driver {self.driver_id} not in idle state")
        
        start_time = time.time()
        
        try:
            # Tokenize
            tokens = self.tokenize(raw_data)
            
            # Create HLLSet via kernel
            hllset = kernel.absorb(tokens)
            
            # Record stats
            duration = time.time() - start_time
            self.stats.record_operation(duration)
            self.ingested_count += 1
            
            # Return to idle
            self.idle()
            
            return hllset
            
        except Exception as e:
            self.mark_error()
            raise RuntimeError(f"Ingest driver {self.driver_id} failed: {e}")
    
    def batch_process(self, raw_data_list: List[str], kernel: Kernel) -> List[HLLSet]:
        """
        Process multiple raw data items in batch.
        
        Returns list of content-addressed HLLSets.
        """
        results = []
        for raw_data in raw_data_list:
            try:
                hllset = self.process(raw_data, kernel)
                results.append(hllset)
            except RuntimeError:
                # Skip failed items, driver will be in ERROR state
                if self.needs_restart:
                    break
        
        return results


# =============================================================================
# SECTION 1: Perceptron - Data Source
# =============================================================================

@dataclass
class Perceptron:
    """
    A perceptron absorbs external reality and produces HLLSets.
    
    Perceptrons are the boundary between external reality and the system.
    They have semantic names (from external reality) but output
    content-addressed HLLSets.
    """
    perceptron_id: str
    source_name: str  # Semantic name from external reality
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    absorption_count: int = 0
    last_absorption: Optional[float] = None
    
    def absorb(self, data: Set[str], kernel: Kernel, p_bits: int = 10) -> HLLSet:
        """
        Absorb data from external reality into HLLSet.
        Uses kernel for transformation.
        """
        hllset = kernel.absorb(data, p_bits=p_bits)
        self.absorption_count += 1
        self.last_absorption = time.time()
        return hllset


# =============================================================================
# SECTION 2: Processing Pipeline Stage
# =============================================================================

@dataclass
class PipelineStage:
    """
    A stage in the processing pipeline.
    
    Each stage transforms HLLSets using kernel operations.
    """
    stage_id: str
    operation: str  # 'union', 'intersection', 'difference', 'filter', etc.
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    processed_count: int = 0
    errors_count: int = 0
    
    def process(self, inputs: List[HLLSet], kernel: Kernel) -> Optional[HLLSet]:
        """
        Process inputs through this stage.
        Returns output HLLSet or None on error.
        """
        if len(inputs) < 1:
            return None
        
        try:
            if self.operation == 'union':
                result = inputs[0]
                for inp in inputs[1:]:
                    result = kernel.union(result, inp)
            elif self.operation == 'intersection':
                result = inputs[0]
                for inp in inputs[1:]:
                    result = kernel.intersection(result, inp)
            elif self.operation == 'difference' and len(inputs) == 2:
                result = kernel.difference(inputs[0], inputs[1])
            elif self.operation == 'identity':
                result = inputs[0]
            else:
                self.errors_count += 1
                return None
            
            self.processed_count += 1
            return result
            
        except Exception as e:
            self.errors_count += 1
            return None


# =============================================================================
# SECTION 3: Persistent State (Git-like)
# =============================================================================

@dataclass
class OSState:
    """
    A snapshot of OS state.
    
    Contains:
    - Root HLLSet (union of all system HLLSets)
    - HRT (Hash Relational Tensor) - memory structure
    - Perceptron states
    - Pipeline configuration
    - Metadata
    
    Stored persistently (e.g., in Git).
    """
    state_hash: str  # SHA1 of this state
    root_hllset_hash: str  # Hash of root HLLSet
    hrt_hash: str  # Hash of HRT
    perceptron_states: Dict[str, Dict[str, Any]]  # perceptron_id -> state
    pipeline_config: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    parent_state: Optional[str] = None  # Hash of parent state
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute state hash from content."""
        content = json.dumps({
            'root_hllset_hash': self.root_hllset_hash,
            'hrt_hash': self.hrt_hash,
            'perceptron_states': self.perceptron_states,
            'pipeline_config': self.pipeline_config,
            'parent_state': self.parent_state,
        }, sort_keys=True)
        return compute_sha1(content)


class PersistentStore:
    """
    Persistent storage for OS states.
    
    Git-like interface:
    - commit: Save current state
    - checkout: Restore state
    - log: View history
    
    Also stores HRTs (Hash Relational Tensors).
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".manifold_storage")
        self.states: Dict[str, OSState] = {}
        self.hrts: Dict[str, HRT] = {}  # HRT hash -> HRT
        self.head: Optional[str] = None
        
        # In-memory cache of HLLSets
        self._hllset_cache: Dict[str, HLLSet] = {}
    
    def commit(self, state: OSState) -> str:
        """
        Commit state to persistent storage.
        Returns state hash.
        """
        state_hash = state.compute_hash()
        state.state_hash = state_hash
        self.states[state_hash] = state
        self.head = state_hash
        
        # Persist to disk if path exists
        if self.storage_path:
            self._persist_state(state)
        
        return state_hash
    
    def checkout(self, state_hash: str) -> Optional[OSState]:
        """Checkout state by hash."""
        if state_hash in self.states:
            self.head = state_hash
            return self.states[state_hash]
        return None
    
    def get_history(self, from_hash: Optional[str] = None) -> List[OSState]:
        """Get state history from given hash back to root."""
        if from_hash is None:
            from_hash = self.head
        
        history = []
        current = from_hash
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            state = self.states.get(current)
            if not state:
                break
            history.append(state)
            current = state.parent_state
        
        return list(reversed(history))
    
    def store_hllset(self, hllset: HLLSet):
        """Store HLLSet in cache."""
        self._hllset_cache[hllset.name] = hllset
    
    def get_hllset(self, hllset_hash: str) -> Optional[HLLSet]:
        """Get HLLSet from cache."""
        return self._hllset_cache.get(hllset_hash)
    
    def store_hrt(self, hrt: HRT) -> str:
        """Store HRT, return its hash."""
        h = hrt.name
        self.hrts[h] = hrt
        return h
    
    def get_hrt(self, hrt_hash: str) -> Optional[HRT]:
        """Get HRT by hash."""
        return self.hrts.get(hrt_hash)
    
    def _persist_state(self, state: OSState):
        """Persist state to disk."""
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save state metadata
        state_file = self.storage_path / f"{state.state_hash}.json"
        with open(state_file, 'w') as f:
            json.dump({
                'state_hash': state.state_hash,
                'root_hllset_hash': state.root_hllset_hash,
                'perceptron_states': state.perceptron_states,
                'pipeline_config': state.pipeline_config,
                'timestamp': state.timestamp,
                'parent_state': state.parent_state,
                'metadata': state.metadata
            }, f, indent=2)
    
    def merge_kernel_cas(self, kernel: Kernel) -> HLLSet:
        """
        Merge kernel CAS into persistent storage.
        
        Since everything is HLLSet and immutable, we can merge
        all HLLSets from kernel CAS into a single root HLLSet.
        """
        hllsets = list(kernel.cas.values())
        if not hllsets:
            # Return empty HLLSet
            return kernel.absorb(set())
        
        # Union all HLLSets from kernel CAS
        root = hllsets[0]
        for hll in hllsets[1:]:
            root = kernel.union(root, hll)
        
        # Store in persistent cache
        self.store_hllset(root)
        
        return root


# =============================================================================
# SECTION 4: Evolution Loop
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for evolution loop."""
    max_iterations: int = 100
    convergence_threshold: float = 0.99
    sleep_interval: float = 0.1
    enable_self_modification: bool = False


class EvolutionLoop:
    """
    Self-generating evolution loop.
    
    The loop:
    1. Ingest data from perceptrons
    2. Process through pipeline
    3. Update system state
    4. Check convergence
    5. Repeat or self-modify
    """
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.iteration_count = 0
        self.convergence_history: List[float] = []
        self.is_running = False
    
    def step(self,
             perceptrons: List[Perceptron],
             pipeline: List[PipelineStage],
             kernel: Kernel,
             store: PersistentStore,
             current_state: Optional[OSState] = None) -> Tuple[OSState, bool]:
        """
        Execute one evolution step.
        
        Returns:
            (new_state, converged)
        """
        self.iteration_count += 1
        
        # 1. Ingest from perceptrons
        ingested_hllsets = []
        for perceptron in perceptrons:
            # In real system, perceptron would have actual data
            # Here we simulate with empty sets
            data = set()  # Would come from external reality
            if data:
                hllset = perceptron.absorb(data, kernel)
                ingested_hllsets.append(hllset)
        
        # 2. Process through pipeline
        processed_hllsets = ingested_hllsets.copy()
        for stage in pipeline:
            if processed_hllsets:
                result = stage.process(processed_hllsets, kernel)
                if result:
                    processed_hllsets = [result]
        
        # 3. Merge with current state
        if current_state:
            current_root = store.get_hllset(current_state.root_hllset_hash)
            if current_root and processed_hllsets:
                new_root = kernel.union(current_root, processed_hllsets[0])
            elif processed_hllsets:
                new_root = processed_hllsets[0]
            elif current_root:
                new_root = current_root
            else:
                new_root = kernel.absorb(set())
        else:
            new_root = processed_hllsets[0] if processed_hllsets else kernel.absorb(set())
        
        # Store new root
        store.store_hllset(new_root)
        
        # 4. Check convergence
        converged = self._check_convergence(new_root, current_state, store)
        
        # 5. Create new state
        new_state = OSState(
            state_hash="",  # Will be computed
            root_hllset_hash=new_root.name,
            perceptron_states={p.perceptron_id: {
                'absorption_count': p.absorption_count,
                'last_absorption': p.last_absorption
            } for p in perceptrons},
            pipeline_config={'stages': [s.stage_id for s in pipeline]},
            parent_state=current_state.state_hash if current_state else None
        )
        
        return new_state, converged
    
    def _check_convergence(self,
                          new_root: HLLSet,
                          current_state: Optional[OSState],
                          store: PersistentStore) -> bool:
        """Check if evolution has converged."""
        if not current_state:
            return False
        
        current_root = store.get_hllset(current_state.root_hllset_hash)
        if not current_root:
            return False
        
        similarity = new_root.similarity(current_root)
        self.convergence_history.append(similarity)
        
        return similarity >= self.config.convergence_threshold
    
    def run(self,
           perceptrons: List[Perceptron],
           pipeline: List[PipelineStage],
           kernel: Kernel,
           store: PersistentStore,
           initial_state: Optional[OSState] = None) -> OSState:
        """
        Run evolution loop until convergence or max iterations.
        """
        self.is_running = True
        current_state = initial_state
        
        for i in range(self.config.max_iterations):
            if not self.is_running:
                break
            
            current_state, converged = self.step(
                perceptrons, pipeline, kernel, store, current_state
            )
            
            if converged:
                print(f"Converged at iteration {i+1}")
                break
            
            time.sleep(self.config.sleep_interval)
        
        self.is_running = False
        return current_state
    
    def stop(self):
        """Stop evolution loop."""
        self.is_running = False


# =============================================================================
# SECTION 5: Manifold OS
# =============================================================================

class ManifoldOS:
    """
    Manifold Operating System - Universal Constructor.
    
    Implements ICASRA pattern:
    - **A (Constructor)**: Validates and commits states (commit/validate)
    - **B (Copier)**: Reproduces states (reproduce via kernel)
    - **C (Controller)**: Manages driver lifecycle and resources
    - **D (Interface)**: Ingests external data via IngestDriver
    
    Orchestrates:
    - Drivers (managed lifecycle: wake, idle, restart, remove)
    - Perceptrons (data sources)
    - Processing pipeline
    - Evolution loop
    - Persistent storage (Git-like)
    - Kernel (stateless transformations)
    - HRT (Hash Relational Tensor) - memory structure
    - Entanglement (lattice-based morphisms)
    
    Design:
    - Immutability: No process can harm another
    - Idempotence: Operations can be repeated safely
    - Content-addressable: Everything identified by hash
    - No scheduling: Pure functional eliminates need
    """
    
    def __init__(self, storage_path: Optional[Path] = None, hrt_config: Optional[HRTConfig] = None):
        self.kernel = Kernel()
        self.store = PersistentStore(storage_path)
        self.evolution = EvolutionLoop()
        
        # HRT configuration (for creating HRTs later)
        self.hrt_config = hrt_config or HRTConfig()
        self.current_hrt: Optional[HRT] = None  # Current HRT
        
        # Entanglement engine
        self.entanglement_engine = EntanglementEngine(self.kernel)
        self.entanglement_morphisms: Dict[str, EntanglementMorphism] = {}  # hrt_hash -> morphism
        
        # Components
        self.perceptrons: Dict[str, Perceptron] = {}
        self.pipeline: List[PipelineStage] = []
        
        # Current state
        self.current_state: Optional[OSState] = None
        
        # Driver management (Universal Constructor pattern)
        self.drivers: Dict[str, Driver] = {}  # driver_id -> Driver
        self.driver_lock = threading.Lock()
        self.driver_health_check_interval = 5.0  # seconds
        self._driver_monitor_thread = None
        self._monitoring_active = False
        
        # Initialize ingest driver
        self._init_ingest_driver()
        
        # Statistics
        self.start_time = time.time()
        self.processing_cycles = 0
    
    def _init_ingest_driver(self):
        """Initialize default ingest driver."""
        driver_id = "ingest_default"
        driver = IngestDriver(driver_id)
        self.register_driver(driver)
        driver.wake()  # Wake driver to idle state
    
    # -------------------------------------------------------------------------
    # Driver Management (Universal Constructor - C Controller)
    # -------------------------------------------------------------------------
    
    def register_driver(self, driver: Driver):
        """
        Register a driver with the OS.
        
        Args:
            driver: Driver instance to register
        """
        with self.driver_lock:
            self.drivers[driver.driver_id] = driver
    
    def unregister_driver(self, driver_id: str) -> bool:
        """
        Unregister and remove a driver.
        
        Args:
            driver_id: ID of driver to remove
        
        Returns:
            True if removed, False if not found
        """
        with self.driver_lock:
            if driver_id in self.drivers:
                driver = self.drivers[driver_id]
                driver.mark_dead()
                del self.drivers[driver_id]
                return True
            return False
    
    def get_driver(self, driver_id: str) -> Optional[Driver]:
        """Get driver by ID."""
        return self.drivers.get(driver_id)
    
    def list_drivers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all drivers with their status.
        
        Returns:
            Dict of driver_id -> {state, type, stats}
        """
        result = {}
        for driver_id, driver in self.drivers.items():
            result[driver_id] = {
                'state': driver.state.value,
                'type': driver.driver_type,
                'is_alive': driver.is_alive,
                'needs_restart': driver.needs_restart,
                'operations': driver.stats.operations_count,
                'errors': driver.stats.errors_count,
                'uptime': time.time() - driver.stats.created_at
            }
        return result
    
    def wake_driver(self, driver_id: str) -> bool:
        """
        Wake a driver from idle state.
        
        Args:
            driver_id: ID of driver to wake
        
        Returns:
            True if woken, False otherwise
        """
        driver = self.get_driver(driver_id)
        if driver:
            return driver.wake()
        return False
    
    def idle_driver(self, driver_id: str) -> bool:
        """
        Put driver to idle state.
        
        Args:
            driver_id: ID of driver to idle
        
        Returns:
            True if idled, False otherwise
        """
        driver = self.get_driver(driver_id)
        if driver:
            return driver.idle()
        return False
    
    def restart_driver(self, driver_id: str) -> bool:
        """
        Restart a driver (error -> idle).
        
        Args:
            driver_id: ID of driver to restart
        
        Returns:
            True if restarted, False otherwise
        """
        driver = self.get_driver(driver_id)
        if driver:
            return driver.restart()
        return False
    
    def cleanup_dead_drivers(self) -> List[str]:
        """
        Remove all dead drivers.
        
        Returns:
            List of removed driver IDs
        """
        removed = []
        with self.driver_lock:
            dead_ids = [
                driver_id for driver_id, driver in self.drivers.items()
                if driver.state == DriverState.DEAD
            ]
            for driver_id in dead_ids:
                del self.drivers[driver_id]
                removed.append(driver_id)
        return removed
    
    def start_driver_monitoring(self):
        """
        Start background thread for driver health monitoring.
        
        Monitors drivers and:
        - Restarts errored drivers
        - Removes dead drivers
        - Puts idle drivers to sleep
        """
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._driver_monitor_thread = threading.Thread(
            target=self._driver_monitor_loop,
            daemon=True
        )
        self._driver_monitor_thread.start()
    
    def stop_driver_monitoring(self):
        """Stop driver monitoring thread."""
        self._monitoring_active = False
        if self._driver_monitor_thread:
            self._driver_monitor_thread.join(timeout=1.0)
    
    def _driver_monitor_loop(self):
        """Background loop for driver health monitoring."""
        while self._monitoring_active:
            try:
                # Check each driver
                for driver_id, driver in list(self.drivers.items()):
                    if not driver.is_alive:
                        # Remove dead driver
                        self.unregister_driver(driver_id)
                    elif driver.needs_restart:
                        # Attempt restart
                        if not driver.restart():
                            # Failed to restart, mark dead
                            driver.mark_dead()
                
                # Cleanup
                self.cleanup_dead_drivers()
                
            except Exception as e:
                print(f"Driver monitor error: {e}")
            
            time.sleep(self.driver_health_check_interval)
    
    # -------------------------------------------------------------------------
    # Ingest Operations (D - Interface)
    # -------------------------------------------------------------------------
    
    def ingest(self, raw_data: str, driver_id: Optional[str] = None) -> Optional[HLLSet]:
        """
        Ingest raw data from external reality.
        
        Args:
            raw_data: Raw string data to ingest
            driver_id: Optional specific driver ID (default: "ingest_default")
        
        Returns:
            Content-addressed HLLSet or None on error
        """
        if driver_id is None:
            driver_id = "ingest_default"
        
        driver = self.get_driver(driver_id)
        if not driver or not isinstance(driver, IngestDriver):
            return None
        
        try:
            hllset = driver.process(raw_data, self.kernel)
            # Store in persistent storage
            self.store.store_hllset(hllset)
            return hllset
        except RuntimeError as e:
            print(f"Ingest failed: {e}")
            return None
    
    def ingest_batch(self, raw_data_list: List[str], 
                    driver_id: Optional[str] = None) -> List[HLLSet]:
        """
        Ingest batch of raw data.
        
        Args:
            raw_data_list: List of raw string data
            driver_id: Optional specific driver ID
        
        Returns:
            List of content-addressed HLLSets
        """
        if driver_id is None:
            driver_id = "ingest_default"
        
        driver = self.get_driver(driver_id)
        if not driver or not isinstance(driver, IngestDriver):
            return []
        
        try:
            hllsets = driver.batch_process(raw_data_list, self.kernel)
            # Store all in persistent storage
            for hllset in hllsets:
                self.store.store_hllset(hllset)
            return hllsets
        except Exception as e:
            print(f"Batch ingest failed: {e}")
            return []

    
    # -------------------------------------------------------------------------
    # Perceptron Management
    # -------------------------------------------------------------------------
    
    def add_perceptron(self, perceptron_id: str, source_name: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Perceptron:
        """Add a perceptron to the system."""
        p = Perceptron(
            perceptron_id=perceptron_id,
            source_name=source_name,
            metadata=metadata or {}
        )
        self.perceptrons[perceptron_id] = p
        return p
    
    def get_perceptron(self, perceptron_id: str) -> Optional[Perceptron]:
        return self.perceptrons.get(perceptron_id)
    
    def list_perceptrons(self) -> List[str]:
        return list(self.perceptrons.keys())
    
    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------
    
    def add_pipeline_stage(self, stage_id: str, operation: str,
                          config: Optional[Dict[str, Any]] = None) -> PipelineStage:
        """Add a stage to the processing pipeline."""
        stage = PipelineStage(
            stage_id=stage_id,
            operation=operation,
            config=config or {}
        )
        self.pipeline.append(stage)
        return stage
    
    def clear_pipeline(self):
        """Clear all pipeline stages."""
        self.pipeline.clear()
    
    # -------------------------------------------------------------------------
    # Processing Cycle
    # -------------------------------------------------------------------------
    
    def process_cycle(self, perceptron_data: Dict[str, Set[str]]) -> OSState:
        """
        Execute one processing cycle.
        
        Args:
            perceptron_data: Dict of perceptron_id -> data to absorb
        
        Returns:
            New OS state
        """
        self.processing_cycles += 1
        
        # 1. Ingest data from perceptrons
        ingested = []
        for perceptron_id, data in perceptron_data.items():
            perceptron = self.perceptrons.get(perceptron_id)
            if perceptron and data:
                hllset = perceptron.absorb(data, self.kernel)
                ingested.append(hllset)
                # Store in kernel CAS
                self.kernel.store(hllset)
        
        # 2. Process through pipeline
        processed = ingested.copy()
        for stage in self.pipeline:
            if processed:
                result = stage.process(processed, self.kernel)
                if result:
                    # Store intermediate result
                    self.kernel.store(result)
                    processed = [result]
        
        # 3. Create HRT from perceptron data
        new_hrt = self.hrt_factory.create_from_perceptrons(perceptron_data, self.kernel)
        
        # 4. Merge HRT with current HRT (if exists)
        if self.current_hrt:
            merged_hrt = new_hrt.merge(self.current_hrt, self.kernel)
            self.current_hrt = merged_hrt
        else:
            self.current_hrt = new_hrt
        
        # Store HRT
        hrt_hash = self.store.store_hrt(self.current_hrt)
        
        # 5. Merge with current state (root HLLSet)
        if self.current_state:
            current_root = self.store.get_hllset(self.current_state.root_hllset_hash)
            if current_root and processed:
                new_root = self.kernel.union(current_root, processed[0])
            elif processed:
                new_root = processed[0]
            elif current_root:
                new_root = current_root
            else:
                new_root = self.kernel.absorb(set())
        else:
            new_root = processed[0] if processed else self.kernel.absorb(set())
        
        # Store new root
        self.store.store_hllset(new_root)
        
        # 6. Create new state (including HRT)
        new_state = OSState(
            state_hash="",
            root_hllset_hash=new_root.name,
            hrt_hash=hrt_hash,
            perceptron_states={
                p.perceptron_id: {
                    'absorption_count': p.absorption_count,
                    'last_absorption': p.last_absorption
                }
                for p in self.perceptrons.values()
            },
            pipeline_config={'stages': [s.stage_id for s in self.pipeline]},
            parent_state=self.current_state.state_hash if self.current_state else None
        )
        
        self.current_state = new_state
        return new_state
    
    # -------------------------------------------------------------------------
    # Commit / Persistence
    # -------------------------------------------------------------------------
    
    def commit(self, message: str = "") -> str:
        """
        Commit current state to persistent storage.
        
        This is the "final commit" that saves OS state before processing.
        Also saves current HRT and state HLLSets.
        """
        if not self.current_state:
            raise RuntimeError("No state to commit")
        
        # Store current state HLLSet before clearing CAS
        state_hll = self.get_root()
        if state_hll:
            self.store.store_hllset(state_hll)
        
        # Merge kernel CAS into persistent storage
        root = self.store.merge_kernel_cas(self.kernel)
        self.current_state.root_hllset_hash = root.name
        
        # Ensure HRT is stored
        if self.current_hrt:
            hrt_hash = self.store.store_hrt(self.current_hrt)
            self.current_state.hrt_hash = hrt_hash
        
        # Commit state
        state_hash = self.store.commit(self.current_state)
        
        # Store the root HLLSet again (in case merge changed it)
        merged_root = self.store.get_hllset(root.name)
        if merged_root:
            self.store.store_hllset(merged_root)
        
        # Clear kernel history (undo/redo no longer needed)
        self.kernel.clear_history()
        
        return state_hash
    
    def checkout(self, state_hash: str) -> Optional[OSState]:
        """Checkout previous state."""
        state = self.store.checkout(state_hash)
        if state:
            self.current_state = state
        return state
    
    def get_history(self) -> List[OSState]:
        """Get full state history."""
        return self.store.get_history()
    
    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    
    def run_evolution(self, max_iterations: int = 100) -> OSState:
        """Run evolution loop."""
        self.evolution.config.max_iterations = max_iterations
        
        final_state = self.evolution.run(
            list(self.perceptrons.values()),
            self.pipeline,
            self.kernel,
            self.store,
            self.current_state
        )
        
        self.current_state = final_state
        return final_state
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def query_similar(self, hllset: HLLSet, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Query HLLSets similar to given one."""
        return self.kernel.find_similar(hllset, threshold)
    
    # -------------------------------------------------------------------------
    # HRT Queries
    # -------------------------------------------------------------------------
    
    def get_hrt(self) -> Optional[HRT]:
        """Get current HRT."""
        return self.current_hrt
    
    def get_hrt_from_state(self, state_hash: Optional[str] = None) -> Optional[HRT]:
        """Get HRT from specific state."""
        if state_hash is None:
            state = self.current_state
        else:
            state = self.store.states.get(state_hash)
        
        if state:
            return self.store.get_hrt(state.hrt_hash)
        return None
    
    def project_future(self, col_indices: List[int]) -> Optional[np.ndarray]:
        """Project columns to rows using HRT (future projection)."""
        if self.current_hrt:
            return self.current_hrt.project_future(col_indices)
        return None
    
    def project_past(self, row_indices: List[int]) -> Optional[np.ndarray]:
        """Project rows to columns using HRT (past reconstruction)."""
        if self.current_hrt:
            return self.current_hrt.project_past(row_indices)
        return None
    
    def compute_cover(self, hllset: HLLSet) -> Optional[Cover]:
        """Compute optimal cover for HLLSet using current HRT lattice."""
        if self.current_hrt and self.current_hrt.get_lattice():
            return self.current_hrt.get_lattice().compute_cover(hllset, self.kernel)
        return None
    
    def get_basic_hllsets(self) -> List[BasicHLLSet]:
        """Get all basic HLLSets from current HRT."""
        if self.current_hrt and self.current_hrt.get_lattice():
            lattice = self.current_hrt.get_lattice()
            return lattice.row_basic + lattice.col_basic
        return []
    
    def get_root(self) -> Optional[HLLSet]:
        """Get current root HLLSet."""
        if self.current_state:
            return self.store.get_hllset(self.current_state.root_hllset_hash)
        return None
    
    # -------------------------------------------------------------------------
    # Entanglement Operations
    # -------------------------------------------------------------------------
    
    def compute_entanglement(self, hrt1_hash: Optional[str] = None, hrt2_hash: Optional[str] = None) -> Optional[EntanglementMorphism]:
        """
        Compute entanglement morphism between two HRTs.
        
        If hashes not provided, uses current HRT and previous HRT from store.
        """
        # Get HRTs
        if hrt1_hash and hrt2_hash:
            hrt1 = self.store.get_hrt(hrt1_hash)
            hrt2 = self.store.get_hrt(hrt2_hash)
        elif self.current_hrt and self.current_state and self.current_state.parent_state:
            # Use current and parent HRT
            hrt2 = self.current_hrt
            parent_state = self.store.states.get(self.current_state.parent_state)
            if parent_state:
                hrt1 = self.store.get_hrt(parent_state.hrt_hash)
            else:
                return None
        else:
            return None
        
        if not hrt1 or not hrt2:
            return None
        
        # Compute entanglement
        morphism = compute_hrt_entanglement(hrt1, hrt2, self.kernel)
        
        # Store morphism
        self.entanglement_morphisms[morphism.name] = morphism
        
        return morphism
    
    def get_entanglement(self, morphism_name: str) -> Optional[EntanglementMorphism]:
        """Get cached entanglement morphism by name."""
        return self.entanglement_morphisms.get(morphism_name)
    
    def list_entanglements(self) -> List[str]:
        """List all entanglement morphism names."""
        return list(self.entanglement_morphisms.keys())
    
    def compute_entanglement_network(self, hrt_hashes: Optional[List[str]] = None) -> Dict[Tuple[str, str], EntanglementMorphism]:
        """
        Compute entanglement network among multiple HRTs.
        
        If hashes not provided, uses all stored HRTs.
        """
        if hrt_hashes:
            hrts = [self.store.get_hrt(h) for h in hrt_hashes]
            hrts = [h for h in hrts if h]
        else:
            hrts = list(self.store.hrts.values())
        
        if len(hrts) < 2:
            return {}
        
        network = compute_entanglement_network(hrts, self.kernel)
        
        # Store all morphisms
        for morphism in network.values():
            self.entanglement_morphisms[morphism.name] = morphism
        
        return network
    
    def find_entangled_pairs(self, hrt_hash: Optional[str] = None, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Find strongly entangled pairs in an HRT.
        
        Returns list of (source_idx, target_idx, strength) tuples.
        """
        if hrt_hash:
            hrt = self.store.get_hrt(hrt_hash)
        else:
            hrt = self.current_hrt
        
        if not hrt:
            return []
        
        # Compute self-entanglement (endomorphism W → W)
        morphism = self.entanglement_engine.compute_self_entanglement(
            hrt.get_lattice(), hrt.name
        )
        
        # Find pairs above threshold
        pairs = []
        for src_idx, matches in morphism.mapping.items():
            for tgt_idx, measurement in matches:
                if measurement.strength >= threshold:
                    pairs.append((src_idx, tgt_idx, measurement.strength))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    # -------------------------------------------------------------------------
    # State-to-State Mapping
    # -------------------------------------------------------------------------
    
    def map_to_previous_state(self) -> Optional[StateMapping]:
        """
        Map current state to previous state using entanglement.
        
        Returns complete mapping including:
        - Decompositions of both states
        - Entanglement morphism
        - Trajectory triple (D, R, N)
        """
        return map_mos_states(self)
    
    def map_states(self, 
                   source_state_hash: Optional[str] = None,
                   target_state_hash: Optional[str] = None) -> Optional[StateMapping]:
        """
        Map any two states.
        
        If hashes not provided, uses current and parent state.
        """
        return map_mos_states(self, source_state_hash, target_state_hash)
    
    def get_trajectory(self) -> Optional[TrajectoryTriple]:
        """Get trajectory triple (D, R, N) from current to parent state."""
        mapping = self.map_to_previous_state()
        if mapping:
            return mapping.trajectory
        return None
    
    def decompose_current_state(self):
        """
        Decompose current state HLLSet into optimal cover.
        
        Returns basic HLLSet indices that cover the state.
        """
        if not self.current_hrt or not self.current_state:
            return None
        
        from .state_mapping import StateMapper
        mapper = StateMapper(self.kernel)
        
        state_hll = self.store.get_hllset(self.current_state.root_hllset_hash)
        if not state_hll:
            return None
        
        decomp = mapper.decompose_state(
            state_hll, 
            self.current_hrt, 
            self.current_state.root_hllset_hash
        )
        
        return decomp
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def stats(self) -> Dict[str, Any]:
        """Get OS statistics."""
        hrt_info = {}
        if self.current_hrt:
            hrt_info = {
                'hrt_name': self.current_hrt.name[:16],
                'hrt_dimension': self.current_hrt.dimension,
                'hrt_am_nonzero': len(self.current_hrt.get_am().get_nonzero_entries()) if self.current_hrt.get_am() else 0,
                'hrt_covers_cached': len(self.current_hrt._covers),
            }
        
        entanglement_info = {
            'entanglement_morphisms': len(self.entanglement_morphisms),
        }
        
        return {
            'runtime': time.time() - self.start_time,
            'processing_cycles': self.processing_cycles,
            'perceptrons': len(self.perceptrons),
            'pipeline_stages': len(self.pipeline),
            'kernel_stats': self.kernel.stats(),
            'persistent_states': len(self.store.states),
            'persistent_hrts': len(self.store.hrts),
            'current_state': self.current_state.state_hash[:16] if self.current_state else None,
            **hrt_info,
            **entanglement_info
        }


# =============================================================================
# SECTION 6: Example Usage
# =============================================================================

def main():
    """Example ManifoldOS usage."""
    print("="*70)
    print("MANIFOLD OS: Evolution Loop and Processing Pipeline")
    print("="*70)
    
    # Create OS
    os = ManifoldOS()
    
    # 1. Add perceptrons
    print("\n1. Adding Perceptrons")
    print("-" * 40)
    
    os.add_perceptron("visual", "camera_01", {"type": "visual", "resolution": "1080p"})
    os.add_perceptron("audio", "microphone_01", {"type": "audio", "sample_rate": 44100})
    os.add_perceptron("tactile", "sensor_01", {"type": "tactile", "sensitivity": "high"})
    
    print(f"Added {len(os.perceptrons)} perceptrons")
    
    # 2. Setup pipeline
    print("\n2. Setup Processing Pipeline")
    print("-" * 40)
    
    os.add_pipeline_stage("merge_sensory", "union")
    os.add_pipeline_stage("refine", "intersection")
    
    print(f"Added {len(os.pipeline)} pipeline stages")
    
    # 3. Process cycle
    print("\n3. Processing Cycle")
    print("-" * 40)
    
    perceptron_data = {
        "visual": {"red", "green", "blue", "yellow"},
        "audio": {"low", "mid", "high", "bass"},
        "tactile": {"soft", "hard", "rough", "smooth"},
    }
    
    state = os.process_cycle(perceptron_data)
    print(f"Processed cycle, new state: {state.state_hash[:16]}...")
    
    root = os.get_root()
    print(f"Root HLLSet: {root}")
    
    # 4. Commit
    print("\n4. Commit to Persistent Storage")
    print("-" * 40)
    
    commit_hash = os.commit("Initial sensory processing")
    print(f"Committed: {commit_hash[:16]}...")
    
    # 5. Stats
    print("\n5. OS Statistics")
    print("-" * 40)
    
    for key, value in os.stats().items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("ManifoldOS ready")
    print("="*70)
    
    return os


if __name__ == "__main__":
    main()
