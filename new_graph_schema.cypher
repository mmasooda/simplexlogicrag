// SimplexRAG New Graph Schema (Neo4j/Cypher)
// Based on ChatGPT-5 recommendations for fire alarm compatibility

// =============================================================================
// CONSTRAINTS AND INDEXES
// =============================================================================

// Node constraints for uniqueness
CREATE CONSTRAINT product_model_unique IF NOT EXISTS
FOR (p:Product) REQUIRE p.model IS UNIQUE;

CREATE CONSTRAINT datasheet_id_unique IF NOT EXISTS
FOR (d:Datasheet) REQUIRE d.doc_id IS UNIQUE;

CREATE CONSTRAINT protocol_name_unique IF NOT EXISTS
FOR (pr:Protocol) REQUIRE pr.name IS UNIQUE;

CREATE CONSTRAINT function_name_unique IF NOT EXISTS
FOR (f:Function) REQUIRE f.name IS UNIQUE;

// Indexes for performance
CREATE INDEX product_category_idx IF NOT EXISTS
FOR (p:Product) ON (p.category);

CREATE INDEX product_family_idx IF NOT EXISTS
FOR (p:Product) ON (p.family);

CREATE INDEX datasheet_title_idx IF NOT EXISTS
FOR (d:Datasheet) ON (d.title);

// =============================================================================
// NODE LABELS AND PROPERTIES
// =============================================================================

// Product node covers heads, bases, panels, modules, accessories
// Properties:
// - model: "4098-9714" (unique identifier)
// - family: "TrueAlarm", "4100ES", etc.
// - category: "Head", "Base", "Panel", "Module", "Accessory"
// - features: ["photoelectric", "heat", "sounder", "isolator", "relay", "co_base"]
// - protocols: ["IDNet", "MAPNET II"]
// - effective_from: date when product became available
// - superseded_by: model that replaces this one (optional)
// - description: human-readable description
// - certifications: ["UL", "FM", "CSFM"]

// Datasheet node for citation and source tracking
// Properties:
// - doc_id: unique document identifier
// - title: document title
// - url: document URL or file path
// - version: document version
// - effective_date: when document was published

// Protocol node for communication protocols
// Properties:
// - name: "IDNet", "MAPNET II"
// - description: protocol description

// Function node for base functions
// Properties:
// - name: "standard", "sounder", "isolator", "relay", "co_base", "multi_sensor"
// - description: function description

// =============================================================================
// RELATIONSHIP TYPES
// =============================================================================

// Core compatibility relationship
// (Head)-[:COMPATIBLE_WITH {notes, source_doc, panel_caveats}]->(Base)
// Properties:
// - notes: additional compatibility notes
// - source_doc: reference to datasheet
// - panel_caveats: ["not 2120 CDT"] - panels that don't support this combination
// - confidence: 0.0-1.0 confidence score

// Function relationships
// (Base)-[:HAS_FUNCTION]->(Function)
// (Product)-[:SUPPORTS_PROTOCOL]->(Protocol)

// Documentation relationships
// (Product)-[:CITED_IN {page, section}]->(Datasheet)

// Supersession relationships
// (OldProduct)-[:SUPERSEDED_BY {effective_date}]->(NewProduct)

// Panel support relationships
// (Panel)-[:SUPPORTS]->(Protocol)
// (Panel)-[:SUPPORTS_FUNCTION]->(Function)

// =============================================================================
// SAMPLE DATA CREATION
// =============================================================================

// Create Protocol nodes
MERGE (idnet:Protocol {name:"IDNet", description:"Simplex IDNet communication protocol"});
MERGE (mapnet:Protocol {name:"MAPNET II", description:"Simplex MAPNET II communication protocol"});

// Create Function nodes
MERGE (std:Function {name:"standard", description:"Standard base with no additional functions"});
MERGE (relay:Function {name:"relay", description:"Base with relay output capabilities"});
MERGE (relay_4w:Function {name:"relay_supervised_4wire", description:"4-wire supervised relay base"});
MERGE (relay_2w:Function {name:"relay_supervised_2wire", description:"2-wire supervised relay base"});
MERGE (sounder:Function {name:"sounder", description:"Base with integrated sounder"});
MERGE (isolator:Function {name:"isolator", description:"Base with circuit isolation"});
MERGE (co_base:Function {name:"co_base", description:"Carbon monoxide sensor base"});
MERGE (co_sounder:Function {name:"co_sounder", description:"CO base with integrated sounder"});
MERGE (multi_sensor:Function {name:"multi_sensor", description:"Multi-sensor base for combined detectors"});
MERGE (multi_sensor_sounder:Function {name:"multi_sensor_sounder", description:"Multi-sensor base with sounder"});

// Create sample Datasheet
MERGE (ds1:Datasheet {
    doc_id:"TrueAlarm_Compatibility_2024", 
    title:"TrueAlarm Detector and Base Compatibility Guide",
    url:"https://simplexgrp.com/datasheets/truealarm_compatibility.pdf",
    version:"2024.1",
    effective_date:"2024-01-01"
});

// =============================================================================
// HEAD PRODUCTS
// =============================================================================

MERGE (h9714:Product {
    model:"4098-9714", 
    family:"TrueAlarm", 
    category:"Head", 
    features:["photoelectric"],
    description:"TrueAlarm Photoelectric Smoke Detector Head",
    certifications:["UL"]
});

MERGE (h9754:Product {
    model:"4098-9754", 
    family:"TrueAlarm", 
    category:"Head", 
    features:["photoelectric","heat"],
    description:"TrueAlarm Photo/Heat Multi-Sensor Detector Head",
    certifications:["UL"]
});

MERGE (h9733:Product {
    model:"4098-9733", 
    family:"TrueAlarm", 
    category:"Head", 
    features:["heat"],
    description:"TrueAlarm Heat Detector Head",
    certifications:["UL"]
});

// Connect heads to protocols
MATCH (h:Product {category:"Head"}), (idnet:Protocol {name:"IDNet"})
MERGE (h)-[:SUPPORTS_PROTOCOL]->(idnet);

// =============================================================================
// BASE PRODUCTS
// =============================================================================

// Standard and relay bases
MERGE (b9792:Product {
    model:"4098-9792", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["standard"],
    description:"Standard Sensor Base (White)",
    certifications:["UL"]
});

MERGE (b9789:Product {
    model:"4098-9789", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["relay","remote_led"],
    description:"Base with connections for remote LED or unsupervised relay",
    certifications:["UL"]
});

MERGE (b9791:Product {
    model:"4098-9791", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["relay_supervised","4-wire"],
    description:"4-wire base for supervised remote relay",
    certifications:["UL"]
});

MERGE (b9780:Product {
    model:"4098-9780", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["relay_supervised","2-wire"],
    description:"2-wire base for supervised remote relay",
    certifications:["UL"]
});

// Sounder base
MERGE (b9794:Product {
    model:"4098-9794", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["sounder"],
    description:"Integrated piezo sounder base",
    certifications:["UL"]
});

// Isolator bases
MERGE (b9793:Product {
    model:"4098-9793", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["isolator"],
    description:"IDNet isolator base (provides SLC isolation)",
    certifications:["UL"]
});

MERGE (b9766:Product {
    model:"4098-9766", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["isolator"],
    description:"Isolator2 base (newer IDNet isolator version)",
    certifications:["UL"]
});

MERGE (b9777:Product {
    model:"4098-9777", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["isolator"],
    description:"Isolator2 base (newer IDNet isolator version)",
    certifications:["UL"]
});

// CO sensor bases
MERGE (b9770:Product {
    model:"4098-9770", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["co_base"],
    description:"CO base (standard carbon monoxide sensor base)",
    certifications:["UL"]
});

MERGE (b9771:Product {
    model:"4098-9771", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["co_sounder"],
    description:"CO sounder base (CO sensor base with integrated sounder)",
    certifications:["UL"]
});

// Multi-sensor bases (specific to 4098-9754)
MERGE (b9796:Product {
    model:"4098-9796", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["multi_sensor"],
    description:"Multi-Sensor standard base that supports two sequential addresses",
    certifications:["UL"]
});

MERGE (b9795:Product {
    model:"4098-9795", 
    family:"TrueAlarm", 
    category:"Base", 
    features:["multi_sensor","sounder"],
    description:"Multi-Sensor base with built-in piezoelectric sounder (88 dBA)",
    certifications:["UL"]
});

// Connect bases to protocols
MATCH (b:Product {category:"Base"}), (idnet:Protocol {name:"IDNet"})
MERGE (b)-[:SUPPORTS_PROTOCOL]->(idnet);

// =============================================================================
// BASE-TO-FUNCTION RELATIONSHIPS
// =============================================================================

// Standard base
MATCH (b:Product {model:"4098-9792"}), (f:Function {name:"standard"})
MERGE (b)-[:HAS_FUNCTION]->(f);

// Relay bases
MATCH (b:Product {model:"4098-9789"}), (f:Function {name:"relay"})
MERGE (b)-[:HAS_FUNCTION]->(f);

MATCH (b:Product {model:"4098-9791"}), (f:Function {name:"relay_supervised_4wire"})
MERGE (b)-[:HAS_FUNCTION]->(f);

MATCH (b:Product {model:"4098-9780"}), (f:Function {name:"relay_supervised_2wire"})
MERGE (b)-[:HAS_FUNCTION]->(f);

// Sounder base
MATCH (b:Product {model:"4098-9794"}), (f:Function {name:"sounder"})
MERGE (b)-[:HAS_FUNCTION]->(f);

// Isolator bases
MATCH (b:Product {model:"4098-9793"}), (f:Function {name:"isolator"})
MERGE (b)-[:HAS_FUNCTION]->(f);

MATCH (b:Product {model:"4098-9766"}), (f:Function {name:"isolator"})
MERGE (b)-[:HAS_FUNCTION]->(f);

MATCH (b:Product {model:"4098-9777"}), (f:Function {name:"isolator"})
MERGE (b)-[:HAS_FUNCTION]->(f);

// CO bases
MATCH (b:Product {model:"4098-9770"}), (f:Function {name:"co_base"})
MERGE (b)-[:HAS_FUNCTION]->(f);

MATCH (b:Product {model:"4098-9771"}), (f:Function {name:"co_sounder"})
MERGE (b)-[:HAS_FUNCTION]->(f);

// Multi-sensor bases
MATCH (b:Product {model:"4098-9796"}), (f:Function {name:"multi_sensor"})
MERGE (b)-[:HAS_FUNCTION]->(f);

MATCH (b:Product {model:"4098-9795"}), (f:Function {name:"multi_sensor_sounder"})
MERGE (b)-[:HAS_FUNCTION]->(f);

// =============================================================================
// HEAD-TO-BASE COMPATIBILITY RELATIONSHIPS (CORRECT DATA)
// =============================================================================

// 4098-9714 (Photoelectric Head) compatibility
MATCH (h:Product {model:"4098-9714"}), (ds:Datasheet {doc_id:"TrueAlarm_Compatibility_2024"})
UNWIND [
    "4098-9792", "4098-9789", "4098-9791", "4098-9780", 
    "4098-9794", "4098-9793", "4098-9766", "4098-9777", 
    "4098-9770", "4098-9771"
] AS base_model
MATCH (b:Product {model:base_model, category:"Base"})
MERGE (h)-[:COMPATIBLE_WITH {
    source_doc:"TrueAlarm_Compatibility_2024",
    confidence:1.0,
    panel_caveats:CASE 
        WHEN base_model IN ["4098-9791","4098-9780"] THEN ["not 2120 CDT"]
        ELSE []
    END
}]->(b);

// 4098-9754 (Photo/Heat Multi-Sensor Head) compatibility
MATCH (h:Product {model:"4098-9754"}), (ds:Datasheet {doc_id:"TrueAlarm_Compatibility_2024"})
UNWIND [
    "4098-9796", "4098-9795", "4098-9792", "4098-9789", 
    "4098-9794", "4098-9791", "4098-9793"
] AS base_model
MATCH (b:Product {model:base_model, category:"Base"})
MERGE (h)-[:COMPATIBLE_WITH {
    source_doc:"TrueAlarm_Compatibility_2024",
    confidence:1.0,
    panel_caveats:CASE 
        WHEN base_model = "4098-9791" THEN ["not 2120 CDT"]
        ELSE []
    END
}]->(b);

// 4098-9733 (Heat Head) compatibility  
MATCH (h:Product {model:"4098-9733"}), (ds:Datasheet {doc_id:"TrueAlarm_Compatibility_2024"})
UNWIND [
    "4098-9792", "4098-9789", "4098-9791", "4098-9780", 
    "4098-9793", "4098-9794", "4098-9770", "4098-9771"
] AS base_model
MATCH (b:Product {model:base_model, category:"Base"})
MERGE (h)-[:COMPATIBLE_WITH {
    source_doc:"TrueAlarm_Compatibility_2024", 
    confidence:1.0,
    panel_caveats:CASE 
        WHEN base_model IN ["4098-9791","4098-9780"] THEN ["not 2120 CDT"]
        ELSE []
    END
}]->(b);

// =============================================================================
// CITATION RELATIONSHIPS
// =============================================================================

// Link all products to the datasheet
MATCH (p:Product), (ds:Datasheet {doc_id:"TrueAlarm_Compatibility_2024"})
MERGE (p)-[:CITED_IN {page:1, section:"Compatibility Matrix"}]->(ds);