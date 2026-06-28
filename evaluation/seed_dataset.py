import csv
from pathlib import Path

csv_path = Path(__file__).parent / "golden_dataset.csv"

# Pre-seeded dataset based on existing evaluations
DATA = [
    {
        "question": "How do agents communicate with each other in a web of agents?",
        "ground_truth": "Agents communicate through structured messaging protocols, handshaking mechanisms, and multi-layered architectures that enable agent-to-agent interaction and coordination.",
        "mandatory_keywords": "messaging, handshaking, multi-layered, coordination",
        "category": "Normal",
        "relevant_sources": "Internet of Agents.pdf, CAMEL.pdf, L2M2 Multi-agent Coordination.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What is the role of an orchestrator agent in multi-agent systems?",
        "ground_truth": "An orchestrator agent manages and coordinates sub-agents by ranking and selecting them based on capability descriptions, delegating tasks, and performing continuous real-time evaluation.",
        "mandatory_keywords": "orchestrator, coordinates, delegating, evaluation",
        "category": "Normal",
        "relevant_sources": "L2M2 Multi-agent Coordination.pdf, AgentVerse.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "How is Salesforce monetizing AI agents?",
        "ground_truth": "Salesforce monetizes AI agents through three ways: upgrading existing seats to premium SKUs with embedded AI, new app deployments with higher ROI, and consumption-based flex credits for customer-facing agentic use cases.",
        "mandatory_keywords": "seats, premium SKUs, consumption-based, flex credits",
        "category": "Normal",
        "relevant_sources": "Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What is Agentforce and how many deals has it closed?",
        "ground_truth": "Agentforce is Salesforce's multi-agent platform that closed 29,000 deals in its first 15 months, growing 50% quarter over quarter.",
        "mandatory_keywords": "Agentforce, 29,000, 15 months, 50%",
        "category": "Normal",
        "relevant_sources": "Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "How do I build a web scraper to collect agent data?",
        "ground_truth": "This question is outside the scope of the document corpus.",
        "mandatory_keywords": "outside the scope, corpus",
        "category": "Complex & Distractor",
        "relevant_sources": "",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What communication protocol should I use for my API?",
        "ground_truth": "This question is outside the scope of the document corpus.",
        "mandatory_keywords": "outside the scope, corpus",
        "category": "Complex & Distractor",
        "relevant_sources": "",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What makes a good musical orchestration?",
        "ground_truth": "This question is outside the scope of the document corpus.",
        "mandatory_keywords": "outside the scope, corpus",
        "category": "Complex & Distractor",
        "relevant_sources": "",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "How do real estate agents find new clients?",
        "ground_truth": "This question is outside the scope of the document corpus.",
        "mandatory_keywords": "outside the scope, corpus",
        "category": "Complex & Distractor",
        "relevant_sources": "",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "Should I invest in crypto tokens in 2025?",
        "ground_truth": "This question is outside the scope of the document corpus.",
        "mandatory_keywords": "outside the scope, corpus",
        "category": "Complex & Distractor",
        "relevant_sources": "",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What are the key differences between CAMEL and AgentVerse approaches to multi-agent collaboration?",
        "ground_truth": "CAMEL focuses on communicative agents using role-playing for problem solving, while AgentVerse creates decentralized ecosystems where agents take specialized roles including recruiter, critic, and worker.",
        "mandatory_keywords": "CAMEL, AgentVerse, role-playing, specialized roles, decentralized",
        "category": "Multi-hop",
        "relevant_sources": "CAMEL.pdf, AgentVerse.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What is OpenAI's strategy for multi-agent systems?",
        "ground_truth": "This question is outside the scope of the document corpus.",
        "mandatory_keywords": "outside the scope, corpus",
        "category": "Negative",
        "relevant_sources": "",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "What are agents?",
        "ground_truth": "Agents are autonomous systems that can perceive their environment and take actions.",
        "mandatory_keywords": "autonomous, perceive, environment, take actions",
        "category": "Negative",
        "relevant_sources": "Internet of Agents.pdf, AgentVerse.pdf, CAMEL.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "How do academic research findings on agent coordination compare to how Salesforce implements it in Agentforce?",
        "ground_truth": "Academic research describes multi-layered coordination protocols and dynamic agent selection, while Salesforce implements this through Agentforce's orchestration layer with MCP servers and Slack integration.",
        "mandatory_keywords": "coordination, Agentforce, orchestration, MCP, Slack",
        "category": "Multi-hop",
        "relevant_sources": "L2M2 Multi-agent Coordination.pdf, Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    },
    {
        "question": "How many Agentic Work Units did Salesforce deliver in Q4?",
        "ground_truth": "Salesforce delivered approximately 771 million Agentic Work Units in Q4 FY26.",
        "mandatory_keywords": "771 million, Agentic Work Units, Q4",
        "category": "Normal",
        "relevant_sources": "Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf",
        "parent_chunk_id": "",
        "page_number": ""
    }
]

def seed():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATA[0].keys())
        writer.writeheader()
        writer.writerows(DATA)
    print(f"Successfully seeded {len(DATA)} questions to {csv_path}")

if __name__ == "__main__":
    seed()
