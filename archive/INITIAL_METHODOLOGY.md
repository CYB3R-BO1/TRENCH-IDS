# TRENCH-IDS Methodology

**Status:** Initial proposed methodology, under review, not yet finalized.

**Research objective:** To investigate whether representations learned from previously seen attack classes help the model learn new, unseen attack classes, and to identify which heterogeneous graph relations contain useful transferable information.

The ten steps below describe the initial proposed approach. They are not all finalized implementation decisions. Each stage will be reviewed, justified, and validated before implementation. The methodology may be revised where necessary to better address the research objective.

Particular attention must be given to task construction, the definition of unseen attack classes, transfer evaluation, and the validation of relation-specific transferability estimates.

1. Raw Traffic Preprocessing
    * Arrange the network traffic into sequential continual learning tasks. The Step 1 task-sequence specification defines the frozen splits, task orders, and evaluation controls; it was approved for Step 1 research-design closure on 2026-11-24. The ten-step proposal remains under review.
    * Each task contains one designated attack class and its deterministically allocated benign traffic. The primary evaluation uses a shared binary attack-versus-benign label space.
    * Preserve chronological order within each task and isolate training, validation, and test windows.
    * Construct 10-second graph windows for primary experiments, 5-second windows for sensitivity, and 30-second windows for exploration. Use temporal bootstrap blocks for uncertainty estimation.
2. Heterogeneous Graph Construction
    * Construct a heterogeneous graph for every graph window.
    * Node types: Host, Flow, Port, Protocol and Service.
    * Relation types: Host→Flow, Flow→Host, Flow→Port, Flow→Protocol, Flow→Service and Host→Host.
    * Initialize each node using the corresponding network traffic features.
3. Relation-Specific Representation Learning
    * Learn separate embeddings for each relation using relation-specific graph encoders.
    * Fuse the relation-specific embeddings through an attention mechanism to generate the final node representation.
    * Store both the relation-specific embeddings and the fused embedding for every node.
4. Attack Classification and Prototype Memory Construction
    * Train the classifier using the fused node embeddings to predict the attack labels of Flow nodes.
    * Compute the mean relation-specific embedding (prototype) for every attack class and every relation.
    * Store the computed prototypes in a Prototype Memory.
5. Warm-up Representation Learning for New Tasks
    * When a new task arrives, repeat graph construction and relation-specific representation learning.
    * Perform a short warm-up training stage (2-3 epochs) to obtain preliminary representations of the new attack classes.
    * Compute temporary relation-specific prototypes for the new attacks.
6. Transferability Estimation
    * Compare the temporary prototypes of the new attacks with all previously stored prototypes.
    * Estimate the transferability of each relation using a similarity measure such as cosine similarity.
    * Identify the previously learned relation-specific representations that are potentially transferable to the new attack classes.
7. Relation Importance Learning
    * Use the transferability scores as input to a lightweight MLP.
    * Learn relation importance weights that indicate how strongly each relation should be preserved during continual learning.
8. Transferability-Guided Continual Learning
    * Continue training the model on the new task using a modified Elastic Weight Consolidation (EWC) objective.
    * Use the learned relation importance weights to adaptively preserve highly transferable relation-specific representations while allowing less transferable relations to adapt.
9. Prototype Memory Update
    * After training the current task, compute the final relation-specific prototypes.
    * Update the Prototype Memory with the newly learned prototypes.
10. Transferability Analysis and Prediction
    * Analyse the final prototypes to determine which learned representations transfer across attack classes.
    * Study relation-wise transferability across successive tasks.
    * For unseen traffic, construct heterogeneous graphs, compute relation-specific embeddings, fuse them using attention and classify the Flow nodes using the trained classifier.
