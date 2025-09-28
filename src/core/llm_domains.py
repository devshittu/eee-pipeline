# src/core/llm_domains.py
# File path: src/core/llm_domains.py
"""
Centralized module for LLM domain classification and few-shot examples.
This improves modularity, keeps the core event_llm_logic clean, and ensures
maximum model performance through dynamic, role-based prompting across 25 domains.
All examples strive to have non-empty fields (events, entities, SOA triplets).
"""
from typing import Dict, Any, List

# --- Domain Classification Keywords ---

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "finance": ["bank", "stock", "market", "economy", "fiscal", "inflation", "interest rate", "invest", "forex"],
    "military": ["military", "conflict", "warship", "frigates", "nato", "defense", "missile", "forces", "army", "navy"],
    "technology": ["AI", "tool", "software", "data", "app", "license", "tech", "model", "cyber", "quantum"],
    "health": ["health", "medical", "disease", "outbreak", "virus", "vaccine", "patient", "pharma", "clinical", "WHO"],
    "politics": ["election", "parliament", "bill", "vote", "cabinet", "policy", "legislation", "senate", "president"],
    "sports": ["league", "game", "match", "cup", "championship", "score", "athlete", "team", "transfer", "doping"],
    "environment": ["climate", "hurricane", "wildfire", "flood", "biodiversity", "conservation", "pollution", "emission", "COP"],
    "science": ["space", "NASA", "CRISPR", "physics", "quantum", "biology", "genetics", "archaeology", "paleontology"],
    "education": ["school", "university", "tuition", "protest", "online learning", "reform", "ranking", "student"],
    "entertainment": ["film", "TV", "streaming", "celebrity", "music", "album", "gossip", "influencer", "award"],
    "arts_culture": ["museum", "exhibition", "art auction", "literary", "theater", "cultural heritage", "preservation"],
    "business": ["merger", "acquisition", "labor strike", "union", "supply chain", "retail", "consumer behavior", "corporate"],
    "law_justice": ["court case", "trial", "legislative change", "human rights", "civil liberties", "police reform", "criminal justice"],
    "international_relations": ["treaty", "summit", "diplomatic visit", "refugee", "migration", "trade agreement", "sanctions", "foreign aid"],
    "religion": ["religious event", "pilgrimage", "interfaith", "church leadership", "papal", "religious freedom"],
    "travel": ["destination opening", "airline industry", "visa policy", "sustainable tourism", "cruise", "airport"],
    "food_agriculture": ["food safety", "farming innovation", "global hunger", "food security", "culinary trends", "restaurant"],
    "transportation": ["public transit", "road construction", "electric vehicle", "aviation", "maritime", "bridge failure"],
    "energy": ["renewable energy", "solar", "wind", "oil and gas", "nuclear energy", "grid reliability", "energy policy"],
    "social_issues": ["gender equality", "LGBTQ+", "racial justice", "discrimination", "housing crisis", "homelessness", "aging population"],
    "crime": ["criminal investigation", "cybercrime", "scams", "community policing", "disaster response"],
    "demographics": ["census data", "population trends", "birth rate", "urbanization", "immigration statistics"],
    "weather": ["seasonal forecast", "record-breaking temperature", "storm", "climate anomalies", "El Niño"],
    "philanthropy": ["donation", "charity", "NGO", "humanitarian aid", "fundraising campaign", "awareness drive"],
    "gaming": ["game release", "esports tournament", "prize pool", "industry controversies", "monetization"]
}

# --- Domain-Specific Few-Shot Examples (Focus on non-empty SOA triplets) ---

DOMAIN_SPECIFIC_EXAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "finance": [
        {
            "input": {"text": "The Central Bank of Europe announced a 25 basis point interest rate hike, leading to a surge in bond yields.", "ner_entities": [{"text": "Central Bank of Europe", "type": "ORG", "start_char": 4, "end_char": 26}]},
            "output": {
                "events": [{"event_type": "economic_policy", "trigger": {"text": "announced", "start_char": 27, "end_char": 36}, "arguments": [{"argument_role": "agent", "entity": {"text": "The Central Bank of Europe", "type": "ORG", "start_char": 0, "end_char": 26}}, {"argument_role": "change", "entity": {"text": "interest rate hike", "type": "OTHER", "start_char": 54, "end_char": 72}}], "metadata": {"sentiment": "neutral", "causality": "The interest rate hike caused a surge in bond yields."}}],
                "extracted_entities": [{"text": "The Central Bank of Europe", "type": "ORG", "start_char": 0, "end_char": 26}],
                "extracted_soa_triplets": [{"subject": {"text": "The Central Bank of Europe", "start_char": 0, "end_char": 26}, "action": {"text": "announced", "start_char": 27, "end_char": 36}, "object": {"text": "a 25 basis point interest rate hike", "start_char": 37, "end_char": 72}}]
            }
        }
    ],
    "military": [
        {
            "input": {"text": "Ukrainian forces launched a counteroffensive near Kharkiv, reclaiming several villages.", "ner_entities": [{"text": "Ukrainian", "type": "NORP", "start_char": 0, "end_char": 9}, {"text": "Kharkiv", "type": "LOC", "start_char": 43, "end_char": 50}]},
            "output": {
                "events": [{"event_type": "military_conflict", "trigger": {"text": "counteroffensive", "start_char": 20, "end_char": 36}, "arguments": [{"argument_role": "agent", "entity": {"text": "Ukrainian forces", "type": "NORP", "start_char": 0, "end_char": 16}}, {"argument_role": "location", "entity": {"text": "near Kharkiv", "type": "LOC", "start_char": 37, "end_char": 49}}], "metadata": {"sentiment": "neutral", "causality": "The counteroffensive led to the reclamation of villages."}}],
                "extracted_entities": [{"text": "Ukrainian forces", "type": "NORP", "start_char": 0, "end_char": 16}],
                "extracted_soa_triplets": [{"subject": {"text": "Ukrainian forces", "start_char": 0, "end_char": 16}, "action": {"text": "launched", "start_char": 17, "end_char": 25}, "object": {"text": "a counteroffensive", "start_char": 26, "end_char": 42}}]
            }
        }
    ],
    "technology": [
        # CRITICAL FIX: The SOA triplet is now fully populated, instructing the LLM to provide it.
        {
            "input": {"text": "The UK government licensed its new AI tool to the US and Canada after a successful pilot.", "ner_entities": [{"text": "UK government", "type": "ORG", "start_char": 4, "end_char": 17}, {"text": "US", "type": "LOC", "start_char": 47, "end_char": 49}, {"text": "Canada", "type": "LOC", "start_char": 54, "end_char": 60}]},
            "output": {
                "events": [{"event_type": "policy_change", "trigger": {"text": "licensed", "start_char": 29, "end_char": 37}, "arguments": [{"argument_role": "agent", "entity": {"text": "UK government", "type": "ORG", "start_char": 4, "end_char": 17}}, {"argument_role": "policy", "entity": {"text": "its new AI tool", "type": "OTHER", "start_char": 38, "end_char": 53}}, {"argument_role": "recipients", "entities": [{"text": "US", "type": "LOC", "start_char": 47, "end_char": 49}, {"text": "Canada", "type": "LOC", "start_char": 54, "end_char": 60}]}], "metadata": {"sentiment": "positive", "causality": "The licensing followed a successful pilot program."}}],
                "extracted_entities": [{"text": "UK government", "type": "ORG", "start_char": 4, "end_char": 17}, {"text": "US", "type": "LOC", "start_char": 47, "end_char": 49}, {"text": "Canada", "type": "LOC", "start_char": 54, "end_char": 60}],
                "extracted_soa_triplets": [
                    {"subject": {"text": "UK government", "start_char": 4, "end_char": 17}, "action": {"text": "licensed",
                                                                                                       "start_char": 29, "end_char": 37}, "object": {"text": "its new AI tool", "start_char": 38, "end_char": 53}}
                ]
            }
        }
    ],
    "health": [
        {
            "input": {"text": "The World Health Organization (WHO) reported a new outbreak of a viral disease in West Africa, prompting a global health alert.", "ner_entities": [{"text": "World Health Organization", "type": "ORG", "start_char": 4, "end_char": 29}, {"text": "WHO", "type": "ORG", "start_char": 31, "end_char": 34}, {"text": "West Africa", "type": "LOC", "start_char": 69, "end_char": 80}]},
            "output": {
                "events": [{"event_type": "disease_outbreak", "trigger": {"text": "outbreak", "start_char": 51, "end_char": 59}, "arguments": [{"argument_role": "agent", "entity": {"text": "World Health Organization", "type": "ORG", "start_char": 4, "end_char": 29}}, {"argument_role": "disease", "entity": {"text": "a viral disease", "type": "MISC", "start_char": 60, "end_char": 75}}, {"argument_role": "location", "entity": {"text": "West Africa", "type": "LOC", "start_char": 69, "end_char": 80}}], "metadata": {"sentiment": "negative", "causality": "The new outbreak prompted a global health alert."}}],
                "extracted_entities": [{"text": "World Health Organization", "type": "ORG", "start_char": 4, "end_char": 29}, {"text": "West Africa", "type": "LOC", "start_char": 69, "end_char": 80}],
                "extracted_soa_triplets": [{"subject": {"text": "World Health Organization", "start_char": 4, "end_char": 29}, "action": {"text": "reported", "start_char": 35, "end_char": 43}, "object": {"text": "a new outbreak of a viral disease", "start_char": 45, "end_char": 76}}]
            }
        }
    ],
    "politics": [
        {
            "input": {"text": "The House of Commons approved the new health spending bill by a vote of 320 to 280, sending it to the Senate.", "ner_entities": [{"text": "House of Commons", "type": "ORG", "start_char": 4, "end_char": 20}, {"text": "Senate", "type": "ORG", "start_char": 75, "end_char": 81}]},
            "output": {
                "events": [{"event_type": "legislation_approval", "trigger": {"text": "approved", "start_char": 21, "end_char": 29}, "arguments": [{"argument_role": "agent", "entity": {"text": "The House of Commons", "type": "ORG", "start_char": 0, "end_char": 20}}, {"argument_role": "legislation", "entity": {"text": "new health spending bill", "type": "LAW", "start_char": 34, "end_char": 57}}, {"argument_role": "target_chamber", "entity": {"text": "Senate", "type": "ORG", "start_char": 75, "end_char": 81}}], "metadata": {"sentiment": "neutral", "causality": "The bill's approval moves it one step closer to becoming law."}}],
                "extracted_entities": [{"text": "House of Commons", "type": "ORG", "start_char": 4, "end_char": 20}, {"text": "Senate", "type": "ORG", "start_char": 75, "end_char": 81}],
                "extracted_soa_triplets": [{"subject": {"text": "The House of Commons", "start_char": 0, "end_char": 20}, "action": {"text": "approved", "start_char": 21, "end_char": 29}, "object": {"text": "the new health spending bill", "start_char": 30, "end_char": 57}}]
            }
        }
    ],
    "sports": [
        {
            "input": {"text": "Liverpool secured a 3-0 victory over Manchester United in the final match, winning the Premier League title.", "ner_entities": [{"text": "Liverpool", "type": "ORG", "start_char": 0, "end_char": 9}, {"text": "Manchester United", "type": "ORG", "start_char": 37, "end_char": 54}, {"text": "Premier League", "type": "ORG", "start_char": 80, "end_char": 94}]},
            "output": {
                "events": [{"event_type": "competition_win", "trigger": {"text": "victory", "start_char": 25, "end_char": 32}, "arguments": [{"argument_role": "winner", "entity": {"text": "Liverpool", "type": "ORG", "start_char": 0, "end_char": 9}}, {"argument_role": "loser", "entity": {"text": "Manchester United", "type": "ORG", "start_char": 37, "end_char": 54}}, {"argument_role": "competition", "entity": {"text": "Premier League", "type": "ORG", "start_char": 80, "end_char": 94}}], "metadata": {"sentiment": "positive", "causality": "The victory secured the Premier League title for Liverpool."}}],
                "extracted_entities": [{"text": "Liverpool", "type": "ORG", "start_char": 0, "end_char": 9}, {"text": "Manchester United", "type": "ORG", "start_char": 37, "end_char": 54}],
                "extracted_soa_triplets": [{"subject": {"text": "Liverpool", "start_char": 0, "end_char": 9}, "action": {"text": "secured", "start_char": 10, "end_char": 17}, "object": {"text": "a 3-0 victory over Manchester United", "start_char": 19, "end_char": 54}}]
            }
        }
    ],
    "environment": [
        {
            "input": {"text": "The UN's climate summit (COP29) resulted in a new international treaty aimed at phasing out plastic pollution by 2040.", "ner_entities": [{"text": "UN", "type": "ORG", "start_char": 4, "end_char": 6}, {"text": "COP29", "type": "EVENT", "start_char": 24, "end_char": 29}, {"text": "2040", "type": "DATE", "start_char": 94, "end_char": 98}]},
            "output": {
                "events": [{"event_type": "policy_agreement", "trigger": {"text": "treaty", "start_char": 52, "end_char": 58}, "arguments": [{"argument_role": "agent", "entity": {"text": "UN", "type": "ORG", "start_char": 4, "end_char": 6}}, {"argument_role": "policy_target", "entity": {"text": "plastic pollution", "type": "OTHER", "start_char": 71, "end_char": 88}}, {"argument_role": "time", "entity": {"text": "2040", "type": "DATE", "start_char": 94, "end_char": 98}}], "metadata": {"sentiment": "positive", "causality": "The climate summit successfully produced a binding international agreement."}}],
                "extracted_entities": [{"text": "UN", "type": "ORG", "start_char": 4, "end_char": 6}, {"text": "COP29", "type": "EVENT", "start_char": 24, "end_char": 29}, {"text": "2040", "type": "DATE", "start_char": 94, "end_char": 98}],
                "extracted_soa_triplets": [{"subject": {"text": "UN's climate summit (COP29)", "start_char": 0, "end_char": 29}, "action": {"text": "resulted", "start_char": 30, "end_char": 38}, "object": {"text": "a new international treaty", "start_char": 44, "end_char": 69}}]
            }
        }
    ],
    "science": [
        {
            "input": {"text": "NASA's Artemis III mission, slated for launch in 2028, will land the first woman and person of color on the Moon.", "ner_entities": [{"text": "NASA", "type": "ORG", "start_char": 0, "end_char": 4}, {"text": "Artemis III", "type": "MISC", "start_char": 7, "end_char": 18}, {"text": "2028", "type": "DATE", "start_char": 39, "end_char": 43}, {"text": "Moon", "type": "LOC", "start_char": 83, "end_char": 87}]},
            "output": {
                "events": [{"event_type": "space_mission", "trigger": {"text": "land", "start_char": 52, "end_char": 56}, "arguments": [{"argument_role": "agent", "entity": {"text": "NASA's Artemis III mission", "type": "ORG", "start_char": 0, "end_char": 28}}, {"argument_role": "target", "entity": {"text": "Moon", "type": "LOC", "start_char": 83, "end_char": 87}}], "metadata": {"sentiment": "positive", "causality": "This mission marks a historical first for gender and racial diversity in space."}}],
                "extracted_entities": [{"text": "NASA", "type": "ORG", "start_char": 0, "end_char": 4}, {"text": "2028", "type": "DATE", "start_char": 39, "end_char": 43}, {"text": "Moon", "type": "LOC", "start_char": 83, "end_char": 87}],
                "extracted_soa_triplets": [{"subject": {"text": "NASA's Artemis III mission", "start_char": 0, "end_char": 28}, "action": {"text": "will land", "start_char": 52, "end_char": 56}, "object": {"text": "the first woman and person of color", "start_char": 57, "end_char": 87}}]
            }
        }
    ],
    "education": [
        {
            "input": {"text": "The University of California board voted to freeze tuition fees for all in-state students through the next two academic years.", "ner_entities": [{"text": "University of California", "type": "ORG", "start_char": 4, "end_char": 28}, {"text": "next two academic years", "type": "DATE", "start_char": 79, "end_char": 102}]},
            "output": {
                "events": [{"event_type": "policy_change", "trigger": {"text": "freeze", "start_char": 45, "end_char": 51}, "arguments": [{"argument_role": "agent", "entity": {"text": "The University of California board", "type": "ORG", "start_char": 0, "end_char": 34}}, {"argument_role": "policy_target", "entity": {"text": "tuition fees", "type": "OTHER", "start_char": 52, "end_char": 64}}, {"argument_role": "time", "entity": {"text": "next two academic years", "type": "DATE", "start_char": 79, "end_char": 102}}], "metadata": {"sentiment": "positive", "causality": "The fee freeze will reduce financial burden on students."}}],
                "extracted_entities": [{"text": "University of California", "type": "ORG", "start_char": 4, "end_char": 28}, {"text": "tuition fees", "type": "OTHER", "start_char": 52, "end_char": 64}],
                "extracted_soa_triplets": [{"subject": {"text": "The University of California board", "start_char": 0, "end_char": 34}, "action": {"text": "voted", "start_char": 35, "end_char": 40}, "object": {"text": "to freeze tuition fees", "start_char": 41, "end_char": 64}}]
            }
        }
    ],
    "entertainment": [
        {
            "input": {"text": "The highly anticipated movie 'Cosmic Dust' shattered global box office records, earning $500 million in its opening weekend.", "ner_entities": [{"text": "Cosmic Dust", "type": "WORK_OF_ART", "start_char": 28, "end_char": 39}, {"text": "$500 million", "type": "MONEY", "start_char": 69, "end_char": 81}]},
            "output": {
                "events": [{"event_type": "record_breaking_performance", "trigger": {"text": "shattered", "start_char": 41, "end_char": 50}, "arguments": [{"argument_role": "subject", "entity": {"text": "Cosmic Dust", "type": "WORK_OF_ART", "start_char": 28, "end_char": 39}}, {"argument_role": "record", "entity": {"text": "global box office records", "type": "OTHER", "start_char": 51, "end_char": 76}}, {"argument_role": "financial_impact", "entity": {"text": "$500 million", "type": "MONEY", "start_char": 69, "end_char": 81}}], "metadata": {"sentiment": "positive", "causality": "Strong positive reception led to massive financial success."}}],
                "extracted_entities": [{"text": "Cosmic Dust", "type": "WORK_OF_ART", "start_char": 28, "end_char": 39}, {"text": "$500 million", "type": "MONEY", "start_char": 69, "end_char": 81}],
                "extracted_soa_triplets": [{"subject": {"text": "The highly anticipated movie 'Cosmic Dust'", "start_char": 0, "end_char": 39}, "action": {"text": "shattered", "start_char": 41, "end_char": 50}, "object": {"text": "global box office records", "start_char": 51, "end_char": 76}}]
            }
        }
    ],
    "arts_culture": [
        {
            "input": {"text": "The Louvre Museum unveiled a newly restored 17th-century Baroque painting by Rembrandt, attracting record crowds.", "ner_entities": [{"text": "Louvre Museum", "type": "ORG", "start_char": 4, "end_char": 17}, {"text": "17th-century", "type": "DATE", "start_char": 37, "end_char": 49}, {"text": "Rembrandt", "type": "PER", "start_char": 72, "end_char": 81}]},
            "output": {
                "events": [{"event_type": "art_exhibition", "trigger": {"text": "unveiled", "start_char": 18, "end_char": 26}, "arguments": [{"argument_role": "agent", "entity": {"text": "The Louvre Museum", "type": "ORG", "start_char": 0, "end_char": 17}}, {"argument_role": "artwork", "entity": {"text": "17th-century Baroque painting", "type": "WORK_OF_ART", "start_char": 37, "end_char": 65}}, {"argument_role": "artist", "entity": {"text": "Rembrandt", "type": "PER", "start_char": 72, "end_char": 81}}], "metadata": {"sentiment": "positive", "causality": "The restoration and unveiling led to a surge in museum attendance."}}],
                "extracted_entities": [{"text": "Louvre Museum", "type": "ORG", "start_char": 4, "end_char": 17}, {"text": "Rembrandt", "type": "PER", "start_char": 72, "end_char": 81}],
                "extracted_soa_triplets": [{"subject": {"text": "The Louvre Museum", "start_char": 0, "end_char": 17}, "action": {"text": "unveiled", "start_char": 18, "end_char": 26}, "object": {"text": "a newly restored 17th-century Baroque painting", "start_char": 27, "end_char": 71}}]
            }
        }
    ],
    "business": [
        {
            "input": {"text": "Acme Corp announced a definitive agreement to acquire Gamma Solutions for $1.5 billion, citing synergy in the logistics sector.", "ner_entities": [{"text": "Acme Corp", "type": "ORG", "start_char": 0, "end_char": 9}, {"text": "Gamma Solutions", "type": "ORG", "start_char": 46, "end_char": 61}, {"text": "$1.5 billion", "type": "MONEY", "start_char": 66, "end_char": 78}]},
            "output": {
                "events": [{"event_type": "merger_acquisition", "trigger": {"text": "acquire", "start_char": 38, "end_char": 45}, "arguments": [{"argument_role": "buyer", "entity": {"text": "Acme Corp", "type": "ORG", "start_char": 0, "end_char": 9}}, {"argument_role": "target", "entity": {"text": "Gamma Solutions", "type": "ORG", "start_char": 46, "end_char": 61}}, {"argument_role": "value", "entity": {"text": "$1.5 billion", "type": "MONEY", "start_char": 66, "end_char": 78}}], "metadata": {"sentiment": "positive", "causality": "The acquisition is driven by anticipated synergy in logistics."}}],
                "extracted_entities": [{"text": "Acme Corp", "type": "ORG", "start_char": 0, "end_char": 9}, {"text": "Gamma Solutions", "type": "ORG", "start_char": 46, "end_char": 61}],
                "extracted_soa_triplets": [{"subject": {"text": "Acme Corp", "start_char": 0, "end_char": 9}, "action": {"text": "announced", "start_char": 10, "end_char": 19}, "object": {"text": "a definitive agreement to acquire Gamma Solutions", "start_char": 22, "end_char": 61}}]
            }
        }
    ],
    "law_justice": [
        {
            "input": {"text": "The Supreme Court delivered a landmark ruling that overturned a state ban on public assembly, strengthening civil liberties.", "ner_entities": [{"text": "Supreme Court", "type": "ORG", "start_char": 4, "end_char": 17}]},
            "output": {
                "events": [{"event_type": "court_ruling", "trigger": {"text": "overturned", "start_char": 40, "end_char": 50}, "arguments": [{"argument_role": "court", "entity": {"text": "The Supreme Court", "type": "ORG", "start_char": 0, "end_char": 17}}, {"argument_role": "target_law", "entity": {"text": "state ban on public assembly", "type": "LAW", "start_char": 53, "end_char": 81}}], "metadata": {"sentiment": "positive", "causality": "The ruling strengthens freedom of speech and assembly."}}],
                "extracted_entities": [{"text": "Supreme Court", "type": "ORG", "start_char": 4, "end_char": 17}],
                "extracted_soa_triplets": [{"subject": {"text": "The Supreme Court", "start_char": 0, "end_char": 17}, "action": {"text": "delivered", "start_char": 18, "end_char": 27}, "object": {"text": "a landmark ruling that overturned a state ban on public assembly", "start_char": 30, "end_char": 81}}]
            }
        }
    ],
    "international_relations": [
        {
            "input": {"text": "France and Germany signed a new trade agreement during a bilateral summit in Berlin yesterday.", "ner_entities": [{"text": "France", "type": "LOC", "start_char": 0, "end_char": 6}, {"text": "Germany", "type": "LOC", "start_char": 11, "end_char": 18}, {"text": "Berlin", "type": "LOC", "start_char": 66, "end_char": 72}, {"text": "yesterday", "type": "DATE", "start_char": 73, "end_char": 82}]},
            "output": {
                "events": [{"event_type": "diplomatic_agreement", "trigger": {"text": "signed", "start_char": 19, "end_char": 25}, "arguments": [{"argument_role": "parties", "entities": [{"text": "France", "type": "LOC", "start_char": 0, "end_char": 6}, {"text": "Germany", "type": "LOC", "start_char": 11, "end_char": 18}]}, {"argument_role": "agreement", "entity": {"text": "new trade agreement", "type": "LAW", "start_char": 32, "end_char": 51}}, {"argument_role": "location", "entity": {"text": "Berlin", "type": "LOC", "start_char": 66, "end_char": 72}}], "metadata": {"sentiment": "positive", "causality": "The agreement is expected to boost bilateral trade."}}],
                "extracted_entities": [{"text": "France", "type": "LOC", "start_char": 0, "end_char": 6}, {"text": "Germany", "type": "LOC", "start_char": 11, "end_char": 18}],
                "extracted_soa_triplets": [{"subject": {"text": "France and Germany", "start_char": 0, "end_char": 18}, "action": {"text": "signed", "start_char": 19, "end_char": 25}, "object": {"text": "a new trade agreement", "start_char": 26, "end_char": 51}}]
            }
        }
    ],
    "religion": [
        {
            "input": {"text": "Pope Francis announced a major change in the Vatican's policy on interfaith dialogue, surprising Catholic leaders worldwide.", "ner_entities": [{"text": "Pope Francis", "type": "PER", "start_char": 0, "end_char": 12}, {"text": "Vatican", "type": "LOC", "start_char": 38, "end_char": 45}, {"text": "Catholic", "type": "NORP", "start_char": 82, "end_char": 90}]},
            "output": {
                "events": [{"event_type": "leadership_announcement", "trigger": {"text": "announced", "start_char": 13, "end_char": 22}, "arguments": [{"argument_role": "agent", "entity": {"text": "Pope Francis", "type": "PER", "start_char": 0, "end_char": 12}}, {"argument_role": "policy", "entity": {"text": "major change in the Vatican's policy on interfaith dialogue", "type": "OTHER", "start_char": 26, "end_char": 81}}], "metadata": {"sentiment": "neutral", "causality": "The announcement will likely reshape global Catholic engagement with other faiths."}}],
                "extracted_entities": [{"text": "Pope Francis", "type": "PER", "start_char": 0, "end_char": 12}, {"text": "Vatican", "type": "LOC", "start_char": 38, "end_char": 45}],
                "extracted_soa_triplets": [{"subject": {"text": "Pope Francis", "start_char": 0, "end_char": 12}, "action": {"text": "announced", "start_char": 13, "end_char": 22}, "object": {"text": "a major change in the Vatican's policy on interfaith dialogue", "start_char": 23, "end_char": 81}}]
            }
        }
    ],
    "travel": [
        {
            "input": {"text": "Japan announced the immediate suspension of its visa-free travel agreement with South Korea due to a sharp rise in cases.", "ner_entities": [{"text": "Japan", "type": "LOC", "start_char": 0, "end_char": 5}, {"text": "South Korea", "type": "LOC", "start_char": 64, "end_char": 75}]},
            "output": {
                "events": [{"event_type": "travel_restriction", "trigger": {"text": "suspension", "start_char": 33, "end_char": 43}, "arguments": [{"argument_role": "agent", "entity": {"text": "Japan", "type": "LOC", "start_char": 0, "end_char": 5}}, {"argument_role": "target_country", "entity": {"text": "South Korea", "type": "LOC", "start_char": 64, "end_char": 75}}], "metadata": {"sentiment": "negative", "causality": "The suspension was a direct response to a sharp rise in disease cases."}}],
                "extracted_entities": [{"text": "Japan", "type": "LOC", "start_char": 0, "end_char": 5}, {"text": "South Korea", "type": "LOC", "start_char": 64, "end_char": 75}],
                "extracted_soa_triplets": [{"subject": {"text": "Japan", "start_char": 0, "end_char": 5}, "action": {"text": "announced", "start_char": 6, "end_char": 15}, "object": {"text": "the immediate suspension of its visa-free travel agreement with South Korea", "start_char": 16, "end_char": 75}}]
            }
        }
    ],
    "food_agriculture": [
        {
            "input": {"text": "The FDA issued a nationwide recall for romaine lettuce after dozens of confirmed E. coli contamination cases were linked to a California farm.", "ner_entities": [{"text": "FDA", "type": "ORG", "start_char": 4, "end_char": 7}, {"text": "E. coli", "type": "MISC", "start_char": 74, "end_char": 81}, {"text": "California", "type": "LOC", "start_char": 113, "end_char": 123}]},
            "output": {
                "events": [{"event_type": "product_recall", "trigger": {"text": "recall", "start_char": 26, "end_char": 32}, "arguments": [{"argument_role": "agent", "entity": {"text": "The FDA", "type": "ORG", "start_char": 0, "end_char": 7}}, {"argument_role": "product", "entity": {"text": "romaine lettuce", "type": "OTHER", "start_char": 37, "end_char": 52}}, {"argument_role": "reason", "entity": {"text": "E. coli contamination", "type": "MISC", "start_char": 74, "end_char": 95}}], "metadata": {"sentiment": "negative", "causality": "Confirmed contamination cases forced the regulatory body to issue the recall."}}],
                "extracted_entities": [{"text": "FDA", "type": "ORG", "start_char": 4, "end_char": 7}, {"text": "romaine lettuce", "type": "OTHER", "start_char": 37, "end_char": 52}],
                "extracted_soa_triplets": [{"subject": {"text": "The FDA", "start_char": 0, "end_char": 7}, "action": {"text": "issued", "start_char": 8, "end_char": 14}, "object": {"text": "a nationwide recall for romaine lettuce", "start_char": 15, "end_char": 52}}]
            }
        }
    ],
    "transportation": [
        {
            "input": {"text": "The city council approved $50 million in funding to expand the subway system's Green Line by six new stations.", "ner_entities": [{"text": "city council", "type": "ORG", "start_char": 4, "end_char": 16}, {"text": "$50 million", "type": "MONEY", "start_char": 27, "end_char": 38}, {"text": "Green Line", "type": "MISC", "start_char": 74, "end_char": 84}]},
            "output": {
                "events": [{"event_type": "infrastructure_development", "trigger": {"text": "expand", "start_char": 53, "end_char": 59}, "arguments": [{"argument_role": "agent", "entity": {"text": "The city council", "type": "ORG", "start_char": 0, "end_char": 16}}, {"argument_role": "funding", "entity": {"text": "$50 million", "type": "MONEY", "start_char": 27, "end_char": 38}}, {"argument_role": "project", "entity": {"text": "subway system's Green Line", "type": "MISC", "start_char": 64, "end_char": 84}}], "metadata": {"sentiment": "positive", "causality": "The funding enables the expansion of public transit services."}}],
                "extracted_entities": [{"text": "city council", "type": "ORG", "start_char": 4, "end_char": 16}, {"text": "$50 million", "type": "MONEY", "start_char": 27, "end_char": 38}],
                "extracted_soa_triplets": [{"subject": {"text": "The city council", "start_char": 0, "end_char": 16}, "action": {"text": "approved", "start_char": 17, "end_char": 25}, "object": {"text": "$50 million in funding to expand the subway system's Green Line", "start_char": 26, "end_char": 84}}]
            }
        }
    ],
    "energy": [
        {
            "input": {"text": "Renewable Energy Corp announced plans to construct a massive new offshore wind farm off the coast of Denmark.", "ner_entities": [{"text": "Renewable Energy Corp", "type": "ORG", "start_char": 0, "end_char": 21}, {"text": "Denmark", "type": "LOC", "start_char": 88, "end_char": 95}]},
            "output": {
                "events": [{"event_type": "project_launch", "trigger": {"text": "construct", "start_char": 37, "end_char": 46}, "arguments": [{"argument_role": "agent", "entity": {"text": "Renewable Energy Corp", "type": "ORG", "start_char": 0, "end_char": 21}}, {"argument_role": "project", "entity": {"text": "offshore wind farm", "type": "OTHER", "start_char": 61, "end_char": 79}}, {"argument_role": "location", "entity": {"text": "Denmark", "type": "LOC", "start_char": 88, "end_char": 95}}], "metadata": {"sentiment": "positive", "causality": "The project will increase clean energy capacity."}}],
                "extracted_entities": [{"text": "Renewable Energy Corp", "type": "ORG", "start_char": 0, "end_char": 21}, {"text": "Denmark", "type": "LOC", "start_char": 88, "end_char": 95}],
                "extracted_soa_triplets": [{"subject": {"text": "Renewable Energy Corp", "start_char": 0, "end_char": 21}, "action": {"text": "announced", "start_char": 22, "end_char": 31}, "object": {"text": "plans to construct a massive new offshore wind farm", "start_char": 32, "end_char": 79}}]
            }
        }
    ],
    "social_issues": [
        {
            "input": {"text": "Advocacy groups praised the city's new ordinance that requires employers to provide paid leave for gender-affirming care.", "ner_entities": [{"text": "Advocacy groups", "type": "ORG", "start_char": 0, "end_char": 15}]},
            "output": {
                "events": [{"event_type": "policy_adoption", "trigger": {"text": "requires", "start_char": 38, "end_char": 46}, "arguments": [{"argument_role": "agent", "entity": {"text": "city's new ordinance", "type": "LAW", "start_char": 27, "end_char": 46}}, {"argument_role": "beneficiary", "entity": {"text": "employees", "type": "PER", "start_char": 55, "end_char": 65}}, {"argument_role": "policy", "entity": {"text": "paid leave for gender-affirming care", "type": "OTHER", "start_char": 70, "end_char": 105}}], "metadata": {"sentiment": "positive", "causality": "The ordinance promotes gender equality and health access."}}],
                "extracted_entities": [{"text": "Advocacy groups", "type": "ORG", "start_char": 0, "end_char": 15}, {"text": "ordinance", "type": "LAW", "start_char": 27, "end_char": 36}],
                "extracted_soa_triplets": [{"subject": {"text": "Advocacy groups", "start_char": 0, "end_char": 15}, "action": {"text": "praised", "start_char": 16, "end_char": 23}, "object": {"text": "the city's new ordinance", "start_char": 24, "end_char": 46}}]
            }
        }
    ],
    "crime": [
        {
            "input": {"text": "Police arrested two suspects linked to the recent bank robbery after a multi-state cybercrime investigation concluded today.", "ner_entities": [{"text": "Police", "type": "ORG", "start_char": 0, "end_char": 6}]},
            "output": {
                "events": [{"event_type": "arrest", "trigger": {"text": "arrested", "start_char": 7, "end_char": 15}, "arguments": [{"argument_role": "agent", "entity": {"text": "Police", "type": "ORG", "start_char": 0, "end_char": 6}}, {"argument_role": "suspects", "entity": {"text": "two suspects", "type": "PER", "start_char": 16, "end_char": 28}}, {"argument_role": "crime", "entity": {"text": "bank robbery", "type": "CRIME", "start_char": 50, "end_char": 62}}], "metadata": {"sentiment": "neutral", "causality": "The arrests concluded the complex, multi-state investigation."}}],
                "extracted_entities": [{"text": "Police", "type": "ORG", "start_char": 0, "end_char": 6}, {"text": "bank robbery", "type": "CRIME", "start_char": 50, "end_char": 62}],
                "extracted_soa_triplets": [{"subject": {"text": "Police", "start_char": 0, "end_char": 6}, "action": {"text": "arrested", "start_char": 7, "end_char": 15}, "object": {"text": "two suspects linked to the recent bank robbery", "start_char": 16, "end_char": 62}}]
            }
        }
    ],
    "demographics": [
        {
            "input": {"text": "New census data revealed that the capital city's population grew by 15% last year due to strong immigration.", "ner_entities": [{"text": "15%", "type": "PERCENT", "start_char": 53, "end_char": 56}, {"text": "last year", "type": "DATE", "start_char": 57, "end_char": 66}]},
            "output": {
                "events": [{"event_type": "population_change", "trigger": {"text": "grew", "start_char": 43, "end_char": 47}, "arguments": [{"argument_role": "subject", "entity": {"text": "the capital city's population", "type": "LOC", "start_char": 28, "end_char": 53}}, {"argument_role": "change_percent", "entity": {"text": "15%", "type": "PERCENT", "start_char": 53, "end_char": 56}}, {"argument_role": "cause", "entity": {"text": "strong immigration", "type": "OTHER", "start_char": 75, "end_char": 93}}], "metadata": {"sentiment": "positive", "causality": "The growth was primarily fueled by new immigrants."}}],
                "extracted_entities": [{"text": "census data", "type": "OTHER", "start_char": 4, "end_char": 15}, {"text": "15%", "type": "PERCENT", "start_char": 53, "end_char": 56}],
                "extracted_soa_triplets": [{"subject": {"text": "New census data", "start_char": 0, "end_char": 15}, "action": {"text": "revealed", "start_char": 16, "end_char": 24}, "object": {"text": "that the capital city's population grew by 15%", "start_char": 25, "end_char": 56}}]
            }
        }
    ],
    "weather": [
        {
            "input": {"text": "Forecasters have issued a rare, level-4 'Extreme Heat' warning for all of the Southwest region starting Friday.", "ner_entities": [{"text": "level-4 'Extreme Heat'", "type": "MISC", "start_char": 32, "end_char": 54}, {"text": "Southwest region", "type": "LOC", "start_char": 68, "end_char": 84}, {"text": "Friday", "type": "DATE", "start_char": 94, "end_char": 100}]},
            "output": {
                "events": [{"event_type": "weather_alert", "trigger": {"text": "warning", "start_char": 55, "end_char": 62}, "arguments": [{"argument_role": "agent", "entity": {"text": "Forecasters", "type": "ORG", "start_char": 0, "end_char": 11}}, {"argument_role": "type", "entity": {"text": "'Extreme Heat'", "type": "MISC", "start_char": 40, "end_char": 54}}, {"argument_role": "location", "entity": {"text": "Southwest region", "type": "LOC", "start_char": 68, "end_char": 84}}], "metadata": {"sentiment": "negative", "causality": "The forecast extreme temperatures necessitated the rare warning."}}],
                "extracted_entities": [{"text": "Forecasters", "type": "ORG", "start_char": 0, "end_char": 11}, {"text": "Southwest region", "type": "LOC", "start_char": 68, "end_char": 84}],
                "extracted_soa_triplets": [{"subject": {"text": "Forecasters", "start_char": 0, "end_char": 11}, "action": {"text": "have issued", "start_char": 12, "end_char": 23}, "object": {"text": "a rare, level-4 'Extreme Heat' warning", "start_char": 24, "end_char": 62}}]
            }
        }
    ],
    "philanthropy": [
        {
            "input": {"text": "The Gates Foundation pledged $500 million over five years to combat malaria in 15 sub-Saharan African nations.", "ner_entities": [{"text": "Gates Foundation", "type": "ORG", "start_char": 4, "end_char": 20}, {"text": "$500 million", "type": "MONEY", "start_char": 29, "end_char": 41}, {"text": "five years", "type": "DATE", "start_char": 47, "end_char": 57}, {"text": "15", "type": "CARDINAL", "start_char": 79, "end_char": 81}, {"text": "African nations", "type": "LOC", "start_char": 94, "end_char": 109}]},
            "output": {
                "events": [{"event_type": "financial_pledge", "trigger": {"text": "pledged", "start_char": 21, "end_char": 28}, "arguments": [{"argument_role": "agent", "entity": {"text": "The Gates Foundation", "type": "ORG", "start_char": 0, "end_char": 20}}, {"argument_role": "amount", "entity": {"text": "$500 million", "type": "MONEY", "start_char": 29, "end_char": 41}}, {"argument_role": "duration", "entity": {"text": "five years", "type": "DATE", "start_char": 47, "end_char": 57}}, {"argument_role": "target_area", "entity": {"text": "15 sub-Saharan African nations", "type": "LOC", "start_char": 79, "end_char": 109}}], "metadata": {"sentiment": "positive", "causality": "The pledge targets a major global health issue."}}],
                "extracted_entities": [{"text": "Gates Foundation", "type": "ORG", "start_char": 4, "end_char": 20}, {"text": "$500 million", "type": "MONEY", "start_char": 29, "end_char": 41}],
                "extracted_soa_triplets": [{"subject": {"text": "The Gates Foundation", "start_char": 0, "end_char": 20}, "action": {"text": "pledged", "start_char": 21, "end_char": 28}, "object": {"text": "$500 million over five years", "start_char": 29, "end_char": 57}}]
            }
        }
    ],
    "gaming": [
        {
            "input": {"text": "Epic Games released its long-awaited battle royale sequel, 'Fortress Royale 2', breaking streaming viewership records on Twitch.", "ner_entities": [{"text": "Epic Games", "type": "ORG", "start_char": 0, "end_char": 10}, {"text": "Fortress Royale 2", "type": "WORK_OF_ART", "start_char": 50, "end_char": 67}, {"text": "Twitch", "type": "ORG", "start_char": 113, "end_char": 119}]},
            "output": {
                "events": [{"event_type": "product_release", "trigger": {"text": "released", "start_char": 11, "end_char": 19}, "arguments": [{"argument_role": "agent", "entity": {"text": "Epic Games", "type": "ORG", "start_char": 0, "end_char": 10}}, {"argument_role": "product", "entity": {"text": "Fortress Royale 2", "type": "WORK_OF_ART", "start_char": 50, "end_char": 67}}, {"argument_role": "platform", "entity": {"text": "Twitch", "type": "ORG", "start_char": 113, "end_char": 119}}], "metadata": {"sentiment": "positive", "causality": "The game's release caused a significant spike in online viewership."}}],
                "extracted_entities": [{"text": "Epic Games", "type": "ORG", "start_char": 0, "end_char": 10}, {"text": "Fortress Royale 2", "type": "WORK_OF_ART", "start_char": 50, "end_char": 67}],
                "extracted_soa_triplets": [{"subject": {"text": "Epic Games", "start_char": 0, "end_char": 10}, "action": {"text": "released", "start_char": 11, "end_char": 19}, "object": {"text": "its long-awaited battle royale sequel, 'Fortress Royale 2'", "start_char": 20, "end_char": 67}}]
            }
        }
    ]
}


def determine_domain(text: str) -> str:
    """
    A simple, fast keyword-based classifier to determine the domain of the text.
    """
    text_lower = text.lower()
    best_match = "general"
    max_keywords = 0

    # Iterate through all keywords to find the domain with the most matches
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        if count > max_keywords:
            max_keywords = count
            best_match = domain

    return best_match


def get_domain_examples(domain: str) -> List[Dict[str, Any]]:
    """
    Returns the few-shot examples for a given domain, defaulting to an empty list.
    """
    # Return the specific domain list if it exists, otherwise return an empty list
    return DOMAIN_SPECIFIC_EXAMPLES.get(domain, [])


def get_domain_persona(domain: str) -> str:
    """
    Returns the role-based persona for a given domain.
    """
    if domain == "finance":
        return "a highly experienced financial news analyst specializing in corporate and economic event extraction"
    elif domain == "military":
        return "a top-tier military intelligence analyst specializing in conflict and defense event extraction"
    elif domain == "technology":
        return "a senior technology journalist specializing in tech product, policy, and market events"
    elif domain == "health":
        return "a veteran public health expert specializing in disease outbreak and health policy events"
    elif domain == "politics":
        return "a seasoned political correspondent specializing in legislative and election event extraction"
    elif domain == "sports":
        return "a dedicated sports reporter specializing in competition and transfer event extraction"
    elif domain == "environment":
        return "a leading climate change expert specializing in environmental policy and natural disaster events"
    elif domain == "science":
        return "a chief science correspondent specializing in research breakthroughs and space exploration events"
    elif domain == "education":
        return "a higher education policy analyst specializing in academic reforms and student activism"
    elif domain == "entertainment":
        return "a senior media and entertainment critic specializing in box office, streaming, and celebrity events"
    elif domain == "arts_culture":
        return "a cultural historian and critic specializing in art acquisitions, literary awards, and cultural heritage"
    elif domain == "business":
        return "a seasoned corporate strategy consultant specializing in M&A, labor issues, and supply chain events"
    elif domain == "law_justice":
        return "a high-profile legal analyst specializing in court rulings, legislative changes, and criminal justice reform"
    elif domain == "international_relations":
        return "a foreign policy expert specializing in diplomatic agreements, summits, and global crises"
    elif domain == "religion":
        return "a religious studies scholar specializing in major spiritual announcements, interfaith dialogue, and institutional changes"
    elif domain == "travel":
        return "a travel industry analyst specializing in destination news, visa policies, and transportation incidents"
    elif domain == "food_agriculture":
        return "a global food security expert specializing in recalls, farming innovations, and culinary trends"
    elif domain == "transportation":
        return "an infrastructure and logistics engineer specializing in public transit, EV development, and construction projects"
    elif domain == "energy":
        return "a utility sector consultant specializing in renewable project launches, market fluctuations, and energy policy"
    elif domain == "social_issues":
        return "a civil rights and public affairs specialist focusing on equality movements, gender rights, and social policy adoption"
    elif domain == "crime":
        return "a criminal investigation expert specializing in major case arrests, cybercrime, and public safety events"
    elif domain == "demographics":
        return "a senior demographic researcher specializing in census data, migration patterns, and population changes"
    elif domain == "weather":
        return "a professional meteorologist specializing in severe weather events, seasonal forecasts, and climate anomalies"
    elif domain == "philanthropy":
        return "a nonprofit sector analyst specializing in major donations, humanitarian aid, and fundraising events"
    elif domain == "gaming":
        return "an esports and gaming industry expert specializing in major releases, tournament results, and industry trends"
    # Fallback persona
    return "an expert event extraction and entity recognition system"

# src/core/llm_domains.py
# File path: src/core/llm_domains.py
