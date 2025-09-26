from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            session.run(
                f"CREATE (n:{label} {{id: $id, label: $label, language: $language}})",
                id=properties["id"],
                label=properties["label"],
                language=properties["language"]
            )
