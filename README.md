# financial-fraud-demo
graph TD
    %% Nodes
    subgraph Storage ["Shared Persistence (Pure Storage)"]
        PVC[("/data Volume")]
    end

    subgraph Pipeline ["5-Tier Pipeline"]
        G[("0. Data Gather<br/>(Job)<br/>No GPU")]
        P[("1. Data Prep<br/>(Job)<br/>2x L40S")]
        B[("2. Model Build<br/>(Job)<br/>2x L40S")]
        I[("3. Inference<br/>(Deployment)<br/>2x L40S")]
        N[("4. Notification<br/>(Deployment)<br/>No GPU")]
    end

    %% Data Flow Edges
    G -->|Writes Raw Data| PVC
    PVC -->|Reads Raw Data| P
    P -->|Writes Features & Graph| PVC
    PVC -->|Reads Features| B
    B -->|Writes Model Artifacts| PVC
    PVC -->|Reads Models| I
    
    %% API Interactions
    I -.->|HTTP Alert (High Fraud Score)| N

    %% Styling
    classDef gpu fill:#76b900,stroke:#333,stroke-width:2px,color:white;
    classDef storage fill:#00A4A6,stroke:#333,stroke-width:2px,color:white;
    class P,B,I gpu;
    class PVC storage;