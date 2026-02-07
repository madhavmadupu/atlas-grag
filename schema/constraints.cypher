// Atlas-GRAG Knowledge Graph Schema
// Supply Chain Risk Analysis Graph Constraints

// =====================
// NODE CONSTRAINTS
// =====================

// Company node - represents organizations in the supply chain
CREATE CONSTRAINT company_name IF NOT EXISTS
FOR (c:Company) REQUIRE c.name IS UNIQUE;

// Product node - represents manufactured products/components  
CREATE CONSTRAINT product_name IF NOT EXISTS
FOR (p:Product) REQUIRE p.name IS UNIQUE;

// Location node - represents geographical locations
CREATE CONSTRAINT location_name IF NOT EXISTS
FOR (l:Location) REQUIRE l.name IS UNIQUE;

// LogisticsNode - represents logistics hubs, ports, warehouses
CREATE CONSTRAINT logistics_node_name IF NOT EXISTS
FOR (n:LogisticsNode) REQUIRE n.name IS UNIQUE;

// RiskEvent node - represents disruption events
CREATE CONSTRAINT risk_event_id IF NOT EXISTS
FOR (r:RiskEvent) REQUIRE r.id IS UNIQUE;


// =====================
// NODE TYPE INDEXES
// =====================

// Speed up label-based lookups
CREATE INDEX company_index IF NOT EXISTS
FOR (c:Company) ON (c.name);

CREATE INDEX product_index IF NOT EXISTS
FOR (p:Product) ON (p.name);

CREATE INDEX location_index IF NOT EXISTS
FOR (l:Location) ON (l.name, l.country);

CREATE INDEX logistics_index IF NOT EXISTS
FOR (n:LogisticsNode) ON (n.name, n.type);

CREATE INDEX risk_event_index IF NOT EXISTS
FOR (r:RiskEvent) ON (r.type, r.severity);


// =====================
// RELATIONSHIP EXAMPLES
// =====================

// (Company)-[:MANUFACTURES]->(Product)
//   Properties: since_year, capacity

// (Company)-[:DEPENDS_ON]->(Company)
//   Properties: dependency_type, criticality

// (Product)-[:STORED_IN]->(Location)
//   Properties: quantity, last_updated

// (Product)-[:COMPONENT_OF]->(Product)
//   Properties: quantity_needed

// (RiskEvent)-[:AFFECTS]->(Location)
//   Properties: impact_severity, start_date, end_date

// (RiskEvent)-[:AFFECTS]->(Company)
//   Properties: impact_severity, start_date, end_date

// (Company)-[:OPERATES_AT]->(Location)
//   Properties: facility_type, capacity

// (LogisticsNode)-[:LOCATED_IN]->(Location)
//   Properties: since_year

// (Company)-[:SHIPS_VIA]->(LogisticsNode)
//   Properties: volume_percentage
